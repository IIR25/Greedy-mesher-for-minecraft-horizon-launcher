//! Greedy meshing + face culling для воксельных чанков.
//!
//! Модуль не зависит от сторонних крейтов и от конкретного движка: на вход
//! подаётся [`PaddedChunk`] (чанк + по одному слою блоков от соседей), на выходе
//! получается [`ChunkMesh`] с компактными вершинами, готовый к загрузке в GPU.
//!
//! # Как это работает
//!
//! 1. **Face culling.** Грань блока создаётся, только если он не воздух и сосед
//!    по этой грани либо воздух, либо прозрачный блок другого типа
//!    (стекло рядом со стеклом внутри себя не рисуется). Граница чанка
//!    обрабатывается корректно за счёт слоя соседних блоков в `PaddedChunk`.
//! 2. **Greedy meshing.** Для каждой оси и каждой плоскости между слоями блоков
//!    строится 2D-маска видимых граней. Затем маска «съедается» прямоугольниками:
//!    сначала растём по ширине, потом по высоте, пока ID блока совпадает.
//!    Один прямоугольник = 4 вершины и 6 индексов вместо N*M граней.
//!
//! # Соглашения
//!
//! * Правосторонняя система координат, Y — вверх.
//! * Передняя сторона треугольника — против часовой стрелки (CCW) при взгляде снаружи.
//! * Позиции вершин — в блоках, относительно угла чанка (`0..=CHUNK_SIZE`).
//!   Умножьте на размер блока и добавьте смещение чанка в вершинном шейдере.
//! * UV — «тайловые»: идут от 0 до размера квада в блоках. Используйте
//!   `fract()` в шейдере (или texture array + `REPEAT`), чтобы текстура
//!   повторялась на каждом блоке объединённого квада. На боковых гранях ось V
//!   направлена вверх.
//!
//! # Пример интеграции
//!
//! ```ignore
//! // Создаются один раз на рабочий поток и переиспользуются (без аллокаций).
//! let mut mesher = GreedyMesher::new();
//! let mut mesh = ChunkMesh::new();
//! let mut padded = PaddedChunk::new();
//!
//! // Заполняем чанк и по одному слою соседей (-1 и CHUNK_SIZE по каждой оси).
//! padded.fill_with(|x, y, z| world.block_at(chunk_origin + ivec3(x, y, z)));
//!
//! // Реестр блоков: достаточно замыкания `Fn(BlockId) -> bool` (true = непрозрачный).
//! mesher.mesh(&padded, &|id| block_registry[id as usize].opaque, &mut mesh);
//!
//! upload_to_gpu(&mesh.vertices, &mesh.indices);
//! ```
//!
//! # Ограничения и расширения
//!
//! Ключ слияния сейчас — только `BlockId`. Если добавите ambient occlusion или
//! освещение по вершинам, включите эти значения в ключ маски: иначе квады с
//! разным освещением будут склеиваться, и градиенты пропадут.

/// Идентификатор блока. `0` зарезервирован под воздух.
pub type BlockId = u16;

/// Воздух: ничего не рисуется и ничего не скрывает.
pub const AIR: BlockId = 0;

/// Размер чанка по каждой оси (в блоках). Позиции вершин хранятся в `u8`,
/// поэтому значение должно быть не больше 255.
pub const CHUNK_SIZE: usize = 16;

const _: () = assert!(
    CHUNK_SIZE >= 1 && CHUNK_SIZE <= 255,
    "CHUNK_SIZE должен лежать в диапазоне 1..=255"
);

/// Размер чанка вместе со слоем соседей с каждой стороны.
const PAD: usize = CHUNK_SIZE + 2;
const PAD_VOLUME: usize = PAD * PAD * PAD;

/// Шаг по индексу массива при движении на 1 блок вдоль оси (X, Y, Z).
/// Раскладка: `x + z * PAD + y * PAD * PAD` (Y — самый «медленный» индекс).
const STRIDE: [usize; 3] = [1, PAD * PAD, PAD];

/// Индекс блока (0, 0, 0) чанка внутри padded-массива.
const ORIGIN: usize = 1 + PAD + PAD * PAD;

// ---------------------------------------------------------------------------
// Публичные типы
// ---------------------------------------------------------------------------

/// Направление грани. Значение (`as u8`) пишется в вершину как индекс нормали.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];

    #[inline]
    pub fn new(axis: usize, negative: bool) -> Face {
        Self::ALL[axis * 2 + negative as usize]
    }

    /// Ось грани: 0 = X, 1 = Y, 2 = Z.
    #[inline]
    pub fn axis(self) -> usize {
        self as usize / 2
    }

    #[inline]
    pub fn is_negative(self) -> bool {
        (self as u8) & 1 == 1
    }

    /// Единичная нормаль грани.
    pub fn normal(self) -> [i32; 3] {
        let mut n = [0; 3];
        n[self.axis()] = if self.is_negative() { -1 } else { 1 };
        n
    }
}

/// Вершина размером 8 байт, `repr(C)` без паддинга — можно напрямую
/// загружать в вершинный буфер (например, через `bytemuck`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vertex {
    /// Позиция в блоках относительно угла чанка, `0..=CHUNK_SIZE`.
    pub position: [u8; 3],
    /// Индекс [`Face`] (он же индекс нормали).
    pub face: u8,
    /// Тайловые UV в блоках: от 0 до размера квада.
    pub uv: [u8; 2],
    /// Тип блока (индекс текстуры/материала).
    pub block: BlockId,
}

/// Результат меширования. Буферы можно очищать и переиспользовать.
#[derive(Default, Debug, Clone)]
pub struct ChunkMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    pub fn new() -> Self {
        Self::default()
    }

    /// Очищает меш, сохраняя выделенную память.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn quad_count(&self) -> usize {
        self.indices.len() / 6
    }

    fn push_quad(&mut self, face: Face, origin: [u8; 3], w: u8, h: u8, block: BlockId) {
        let (u, v) = tangent_axes(face.axis());
        let base = self.vertices.len() as u32;

        // Порядок углов (u, v): (0,0) -> (w,0) -> (w,h) -> (0,h).
        // Так как e_u x e_v = +e_axis, этот порядок CCW для граней +axis.
        for (du, dv) in [(0, 0), (w, 0), (w, h), (0, h)] {
            let mut position = origin;
            position[u] += du;
            position[v] += dv;
            self.vertices.push(Vertex {
                position,
                face: face as u8,
                uv: face_uv(face, du, dv, w, h),
                block,
            });
        }

        // Для граней -axis разворачиваем порядок обхода.
        let order: [u32; 6] = if face.is_negative() {
            [0, 2, 1, 0, 3, 2]
        } else {
            [0, 1, 2, 0, 2, 3]
        };
        self.indices.extend(order.iter().map(|&i| base + i));
    }
}

/// Информация о блоках, нужная мешеру. Для быстрой интеграции достаточно
/// передать замыкание `|id| ...` (оно реализует этот трейт автоматически).
pub trait BlockRegistry {
    /// `true`, если блок полностью скрывает грани за собой (камень, земля).
    /// `false` для стекла, листвы, воды и т. п. Для [`AIR`] не вызывается.
    fn is_opaque(&self, id: BlockId) -> bool;
}

impl<F: Fn(BlockId) -> bool> BlockRegistry for F {
    #[inline]
    fn is_opaque(&self, id: BlockId) -> bool {
        self(id)
    }
}

/// Чанк `CHUNK_SIZE³` плюс слой соседних блоков вокруг (координаты `-1..=CHUNK_SIZE`).
///
/// Слой соседей нужен, чтобы скрывать грани на границе чанков без обращений
/// к миру во время меширования. Если соседний чанк ещё не загружен, оставьте
/// там `AIR`: грани у границы будут нарисованы и позже перестроены.
#[derive(Clone)]
pub struct PaddedChunk {
    blocks: Box<[BlockId]>,
}

impl Default for PaddedChunk {
    fn default() -> Self {
        Self::new()
    }
}

impl PaddedChunk {
    pub fn new() -> Self {
        Self {
            blocks: vec![AIR; PAD_VOLUME].into_boxed_slice(),
        }
    }

    #[inline]
    fn index(x: i32, y: i32, z: i32) -> usize {
        debug_assert!(
            [x, y, z]
                .iter()
                .all(|&c| (-1..=CHUNK_SIZE as i32).contains(&c)),
            "координаты ({x}, {y}, {z}) вне диапазона -1..=CHUNK_SIZE"
        );
        (x + 1) as usize + (z + 1) as usize * PAD + (y + 1) as usize * PAD * PAD
    }

    /// Блок по локальным координатам (`-1..=CHUNK_SIZE`, включая слой соседей).
    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> BlockId {
        self.blocks[Self::index(x, y, z)]
    }

    #[inline]
    pub fn set(&mut self, x: i32, y: i32, z: i32, id: BlockId) {
        self.blocks[Self::index(x, y, z)] = id;
    }

    /// Заполняет весь объём (вместе со слоем соседей) одним блоком.
    pub fn fill(&mut self, id: BlockId) {
        self.blocks.fill(id);
    }

    /// Заполняет весь объём (вместе со слоем соседей) значениями из функции.
    /// Координаты передаются в диапазоне `-1..=CHUNK_SIZE`.
    pub fn fill_with(&mut self, mut f: impl FnMut(i32, i32, i32) -> BlockId) {
        let n = CHUNK_SIZE as i32;
        for y in -1..=n {
            for z in -1..=n {
                for x in -1..=n {
                    self.set(x, y, z, f(x, y, z));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Мешер
// ---------------------------------------------------------------------------

/// Строит меш чанка. Хранит рабочие буферы масок, поэтому его нужно создать
/// один раз (на поток) и вызывать [`GreedyMesher::mesh`] многократно:
/// в горячем пути аллокаций нет.
pub struct GreedyMesher {
    /// Маска граней, смотрящих в +axis. Значение — ID владельца грани, `AIR` = грани нет.
    front: Vec<BlockId>,
    /// Маска граней, смотрящих в -axis.
    back: Vec<BlockId>,
}

impl Default for GreedyMesher {
    fn default() -> Self {
        Self::new()
    }
}

impl GreedyMesher {
    pub fn new() -> Self {
        Self {
            front: vec![AIR; CHUNK_SIZE * CHUNK_SIZE],
            back: vec![AIR; CHUNK_SIZE * CHUNK_SIZE],
        }
    }

    /// Строит меш для `chunk` и записывает его в `out` (старое содержимое
    /// `out` удаляется, память переиспользуется).
    pub fn mesh<R: BlockRegistry>(
        &mut self,
        chunk: &PaddedChunk,
        registry: &R,
        out: &mut ChunkMesh,
    ) {
        out.clear();

        for axis in 0..3 {
            // Плоскость `slice` лежит между слоями блоков `slice - 1` и `slice`.
            for slice in 0..=CHUNK_SIZE {
                let (any_front, any_back) = build_masks(
                    chunk,
                    registry,
                    axis,
                    slice,
                    &mut self.front,
                    &mut self.back,
                );
                // Пустые плоскости (самый частый случай) пропускаем целиком.
                if any_front {
                    greedy_merge(&mut self.front, Face::new(axis, false), slice, out);
                }
                if any_back {
                    greedy_merge(&mut self.back, Face::new(axis, true), slice, out);
                }
            }
        }
    }
}

/// Удобная обёртка для разового вызова. В цикле по чанкам используйте
/// [`GreedyMesher`], чтобы не аллоцировать буферы каждый раз.
pub fn mesh_chunk<R: BlockRegistry>(chunk: &PaddedChunk, registry: &R) -> ChunkMesh {
    let mut out = ChunkMesh::new();
    GreedyMesher::new().mesh(chunk, registry, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Внутренняя кухня
// ---------------------------------------------------------------------------

/// Две оси, лежащие в плоскости грани. `(axis, u, v)` — циклическая перестановка
/// (X, Y, Z), поэтому `e_u x e_v = +e_axis`.
#[inline]
fn tangent_axes(axis: usize) -> (usize, usize) {
    ((axis + 1) % 3, (axis + 2) % 3)
}

/// Face culling: видна ли грань блока `owner`, если с этой стороны стоит `neighbor`.
#[inline(always)]
fn face_visible<R: BlockRegistry>(owner: BlockId, neighbor: BlockId, registry: &R) -> bool {
    owner != AIR && (neighbor == AIR || (owner != neighbor && !registry.is_opaque(neighbor)))
}

/// Строит маски видимых граней для плоскости `slice` по оси `axis`.
/// Возвращает, есть ли хоть одна грань в `front` и в `back`.
fn build_masks<R: BlockRegistry>(
    chunk: &PaddedChunk,
    registry: &R,
    axis: usize,
    slice: usize,
    front: &mut [BlockId],
    back: &mut [BlockId],
) -> (bool, bool) {
    let (u, v) = tangent_axes(axis);
    let (stride_d, stride_u, stride_v) = (STRIDE[axis], STRIDE[u], STRIDE[v]);
    let plane = ORIGIN + slice * stride_d;
    let blocks = &chunk.blocks[..];

    let (mut any_front, mut any_back) = (false, false);

    for j in 0..CHUNK_SIZE {
        let row = plane + j * stride_v;
        for i in 0..CHUNK_SIZE {
            let idx = row + i * stride_u;
            let behind = blocks[idx - stride_d]; // блок со стороны -axis
            let ahead = blocks[idx]; // блок со стороны +axis
            let cell = j * CHUNK_SIZE + i;

            // Одинаковые блоки (в т. ч. воздух рядом с воздухом) граней не дают.
            if behind == ahead {
                front[cell] = AIR;
                back[cell] = AIR;
                continue;
            }

            front[cell] = if face_visible(behind, ahead, registry) {
                any_front = true;
                behind
            } else {
                AIR
            };
            back[cell] = if face_visible(ahead, behind, registry) {
                any_back = true;
                ahead
            } else {
                AIR
            };
        }
    }

    (any_front, any_back)
}

/// Жадно объединяет ячейки маски в прямоугольники и добавляет их в `out`.
/// Маска при этом обнуляется, так что её можно сразу использовать повторно.
fn greedy_merge(mask: &mut [BlockId], face: Face, slice: usize, out: &mut ChunkMesh) {
    let axis = face.axis();
    let (u, v) = tangent_axes(axis);

    for j in 0..CHUNK_SIZE {
        let mut i = 0;
        while i < CHUNK_SIZE {
            let block = mask[j * CHUNK_SIZE + i];
            if block == AIR {
                i += 1;
                continue;
            }

            // Ширина: растём вдоль оси u, пока ID совпадает.
            let mut w = 1;
            while i + w < CHUNK_SIZE && mask[j * CHUNK_SIZE + i + w] == block {
                w += 1;
            }

            // Высота: растём вдоль оси v, пока вся полоса шириной w совпадает.
            let mut h = 1;
            while j + h < CHUNK_SIZE {
                let row = (j + h) * CHUNK_SIZE + i;
                if mask[row..row + w].iter().any(|&c| c != block) {
                    break;
                }
                h += 1;
            }

            // Помечаем использованные ячейки.
            for dj in 0..h {
                let row = (j + dj) * CHUNK_SIZE + i;
                mask[row..row + w].fill(AIR);
            }

            let mut origin = [0u8; 3];
            origin[axis] = slice as u8;
            origin[u] = i as u8;
            origin[v] = j as u8;
            out.push_quad(face, origin, w as u8, h as u8, block);

            i += w;
        }
    }
}

/// UV для угла квада со смещением (`du`, `dv`) вдоль осей (u, v), квад `w` x `h`.
/// Подобрано так, чтобы на боковых гранях V смотрела вверх, а текстура не
/// отражалась зеркально при взгляде снаружи.
#[inline]
fn face_uv(face: Face, du: u8, dv: u8, w: u8, h: u8) -> [u8; 2] {
    match face {
        Face::PosX => [h - dv, du],
        Face::NegX => [dv, du],
        Face::PosZ => [du, dv],
        Face::NegZ => [w - du, dv],
        Face::PosY | Face::NegY => [dv, du],
    }
}

// ---------------------------------------------------------------------------
// Тесты
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const N: i32 = CHUNK_SIZE as i32;
    const STONE: BlockId = 1;
    const DIRT: BlockId = 2;
    const GLASS: BlockId = 3;

    fn registry(id: BlockId) -> bool {
        id == STONE || id == DIRT
    }

    fn mesh(chunk: &PaddedChunk) -> ChunkMesh {
        let mut out = ChunkMesh::new();
        GreedyMesher::new().mesh(chunk, &registry, &mut out);
        out
    }

    fn fill_box(chunk: &mut PaddedChunk, min: [i32; 3], max: [i32; 3], id: BlockId) {
        for y in min[1]..max[1] {
            for z in min[2]..max[2] {
                for x in min[0]..max[0] {
                    chunk.set(x, y, z, id);
                }
            }
        }
    }

    /// Детерминированный генератор случайных чанков (LCG), включая слой соседей.
    fn pseudo_random_chunk(seed: u64, solid_percent: u64) -> PaddedChunk {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };
        let mut chunk = PaddedChunk::new();
        chunk.fill_with(|_, _, _| {
            if next() % 100 < solid_percent {
                1 + (next() % 3) as BlockId
            } else {
                AIR
            }
        });
        chunk
    }

    /// Эталон: наивный перебор всех блоков и всех 6 направлений.
    fn brute_force_faces(chunk: &PaddedChunk) -> HashSet<(u8, i32, i32, i32)> {
        let mut set = HashSet::new();
        for y in 0..N {
            for z in 0..N {
                for x in 0..N {
                    let owner = chunk.get(x, y, z);
                    for face in Face::ALL {
                        let n = face.normal();
                        let neighbor = chunk.get(x + n[0], y + n[1], z + n[2]);
                        let visible = owner != AIR
                            && (neighbor == AIR || (owner != neighbor && !registry(neighbor)));
                        if visible {
                            set.insert((face as u8, x, y, z));
                        }
                    }
                }
            }
        }
        set
    }

    /// Какие единичные грани покрывает меш. Падает, если грань покрыта дважды.
    fn covered_faces(m: &ChunkMesh) -> HashSet<(u8, i32, i32, i32)> {
        let mut set = HashSet::new();
        for quad in m.vertices.chunks(4) {
            let face = Face::ALL[quad[0].face as usize];
            let axis = face.axis();
            let (u, v) = tangent_axes(axis);
            let lo = |a: usize| quad.iter().map(|vx| vx.position[a] as i32).min().unwrap();
            let hi = |a: usize| quad.iter().map(|vx| vx.position[a] as i32).max().unwrap();

            let plane = lo(axis);
            assert_eq!(plane, hi(axis), "квад должен быть плоским");
            let owner_layer = if face.is_negative() { plane } else { plane - 1 };

            for a in lo(u)..hi(u) {
                for b in lo(v)..hi(v) {
                    let mut p = [0i32; 3];
                    p[axis] = owner_layer;
                    p[u] = a;
                    p[v] = b;
                    assert!(
                        set.insert((face as u8, p[0], p[1], p[2])),
                        "грань {p:?} покрыта дважды"
                    );
                }
            }
        }
        set
    }

    #[test]
    fn empty_chunk_has_no_geometry() {
        assert!(mesh(&PaddedChunk::new()).is_empty());
    }

    #[test]
    fn single_block_has_six_quads() {
        let mut chunk = PaddedChunk::new();
        chunk.set(5, 5, 5, STONE);
        let m = mesh(&chunk);
        assert_eq!(m.quad_count(), 6);
        assert_eq!(m.vertices.len(), 24);
        assert_eq!(m.indices.len(), 36);
    }

    #[test]
    fn solid_chunk_collapses_to_six_quads() {
        let mut chunk = PaddedChunk::new();
        fill_box(&mut chunk, [0, 0, 0], [N, N, N], STONE);
        assert_eq!(mesh(&chunk).quad_count(), 6);
    }

    #[test]
    fn flat_floor_is_six_quads() {
        let mut chunk = PaddedChunk::new();
        fill_box(&mut chunk, [0, 0, 0], [N, 1, N], STONE);
        // верх, низ и 4 боковые стороны
        assert_eq!(mesh(&chunk).quad_count(), 6);
    }

    #[test]
    fn neighbor_layer_culls_border_faces() {
        let mut chunk = PaddedChunk::new();
        chunk.fill(STONE); // и сам чанк, и все соседи сплошные
        assert!(mesh(&chunk).is_empty());
    }

    #[test]
    fn shared_faces_are_culled_and_same_blocks_merge() {
        let mut chunk = PaddedChunk::new();
        chunk.set(3, 3, 3, STONE);
        chunk.set(4, 3, 3, STONE);
        // 2 торца + по одной объединённой грани с остальных 4 сторон
        assert_eq!(mesh(&chunk).quad_count(), 6);

        let mut chunk = PaddedChunk::new();
        chunk.set(3, 3, 3, STONE);
        chunk.set(4, 3, 3, DIRT);
        // 2 торца + по 2 раздельных грани (разные блоки) с 4 сторон
        assert_eq!(mesh(&chunk).quad_count(), 10);
    }

    #[test]
    fn transparent_blocks_are_handled() {
        let mut chunk = PaddedChunk::new();
        chunk.set(3, 3, 3, GLASS);
        chunk.set(4, 3, 3, GLASS);
        // грань между одинаковыми стёклами скрыта
        assert_eq!(mesh(&chunk).quad_count(), 6);

        let mut chunk = PaddedChunk::new();
        chunk.set(3, 3, 3, GLASS);
        chunk.set(4, 3, 3, STONE);
        // стекло: 5 граней (к камню скрыта), камень: 6 (к стеклу видна)
        assert_eq!(mesh(&chunk).quad_count(), 11);
    }

    #[test]
    fn matches_naive_face_culling_on_random_chunks() {
        for seed in 0..40u64 {
            for &density in &[5u64, 30, 60, 95] {
                let chunk = pseudo_random_chunk(seed, density);
                let m = mesh(&chunk);
                assert_eq!(
                    covered_faces(&m),
                    brute_force_faces(&chunk),
                    "seed {seed}, density {density}"
                );
            }
        }
    }

    #[test]
    fn triangles_face_outward_and_indices_are_valid() {
        let chunk = pseudo_random_chunk(7, 50);
        let m = mesh(&chunk);
        assert_eq!(m.indices.len() % 6, 0);

        for tri in m.indices.chunks(3) {
            assert!(tri.iter().all(|&i| (i as usize) < m.vertices.len()));

            let p = |k: usize| {
                let pos = m.vertices[tri[k] as usize].position;
                [pos[0] as i32, pos[1] as i32, pos[2] as i32]
            };
            let (a, b, c) = (p(0), p(1), p(2));
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];

            let face = Face::ALL[m.vertices[tri[0] as usize].face as usize];
            let n = face.normal();
            let dot = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2];
            assert!(dot > 0, "треугольник смотрит внутрь: {face:?}");
        }
    }

    #[test]
    fn mesher_can_be_reused_without_state_leaks() {
        let a = pseudo_random_chunk(1, 40);
        let b = pseudo_random_chunk(2, 70);

        let mut reused = GreedyMesher::new();
        let mut out = ChunkMesh::new();
        reused.mesh(&a, &registry, &mut out);
        reused.mesh(&b, &registry, &mut out);

        let fresh = mesh(&b);
        assert_eq!(out.vertices, fresh.vertices);
        assert_eq!(out.indices, fresh.indices);
    }

    #[test]
    fn tiled_uvs_match_quad_size() {
        let mut chunk = PaddedChunk::new();
        fill_box(&mut chunk, [0, 0, 0], [4, 1, 3], STONE);
        let m = mesh(&chunk);
        assert!(m.vertices.iter().all(|v| v.uv[0] <= 4 && v.uv[1] <= 4));
        // у верхней грани 4x3 блока максимальные UV равны размерам квада
        let top: Vec<_> = m
            .vertices
            .iter()
            .filter(|v| v.face == Face::PosY as u8)
            .collect();
        assert_eq!(top.len(), 4);
        assert_eq!(top.iter().map(|v| v.uv[0]).max(), Some(4)); // вдоль X
        assert_eq!(top.iter().map(|v| v.uv[1]).max(), Some(3)); // вдоль Z
    }
}

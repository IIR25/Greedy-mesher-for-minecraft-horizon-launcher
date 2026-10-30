pub mod voxel;

use voxel::greedy_mesh::{ChunkMesh, GreedyMesher, PaddedChunk};

fn main() {
    println!("Инициализация воксельного движка Horizon...");

    let mut mesher = GreedyMesher::new();
    let mut mesh = ChunkMesh::new();
    let mut padded = PaddedChunk::new();

    // Создадим небольшой кубик блоков в центре чанка (например, от 4 до 12 по всем осям)
    for y in 4..12 {
        for z in 4..12 {
            for x in 4..12 {
                padded.set(x, y, z, 1); // 1 — это ID камня
            }
        }
    }

    // Реестр блоков: передаем замыкание, где ID != 0 считаются непрозрачными
    let registry = |id: voxel::greedy_mesh::BlockId| id != 0;

    // Генерируем оптимизированную сетку чанка
    mesher.mesh(&padded, &registry, &mut mesh);

    println!("Успешно сгенерировано квадов (граней): {}", mesh.quad_count());
    println!("Вершин в буфере: {}", mesh.vertices.len());
}
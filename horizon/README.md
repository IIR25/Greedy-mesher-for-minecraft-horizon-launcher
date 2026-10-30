\# Horizon Greedy Mesher



A high-performance, zero-dependency Rust library designed for optimized voxel chunk mesh generation using Greedy Meshing algorithms. Developed for seamless integration with the Horizon voxel engine and wgpu graphics pipelines.



\## Features



\* Greedy Meshing Algorithm: Combines adjacent coplanar voxel faces to drastically reduce total vertex counts and improve GPU render performance.

\* Face Culling: Automatically skips hidden inner faces between adjacent solid blocks.

\* wgpu \& bytemuck Integration: Layout is fully #\[repr(C)] compliant for zero-cost GPU buffer uploads.

\* Native Plugin Architecture: Compiles to a shared dynamic library (cdylib / .dll / .so) for direct loading into the Horizon engine.



\## Usage \& Integration



Add the compiled .dll / .so library into your Horizon mods/ directory, or include the crate directly in your Rust voxel pipeline:



```rust

use wgpu::util::DeviceExt;



// Fast GPU buffer initialization via bytemuck

let vertex\_buffer = device.create\_buffer\_init(\&wgpu::util::BufferInitDescriptor {

&#x20;   label: Some("Chunk Vertex Buffer"),

&#x20;   contents: bytemuck::cast\_slice(\&mesh.vertices),

&#x20;   usage: wgpu::BufferUsages::VERTEX,

});


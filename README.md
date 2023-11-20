[![builds.sr.ht status](https://builds.sr.ht/~gigablaster.svg)](https://builds.sr.ht/~gigablaster?)

My own vulkan-based game engine. I'm trying to implement everything in way I want it to work based on my more than 10 years
of experience in game development.

# Features
- Written on Rust from start to finish.
- Data-orientired design.
- Vulkan 1.3, fully based on VK_KHR_dynamic_rendering.
- Some degreee of multithreading, as much as I need.
- Handle-based API, HypeHype-style (https://www.youtube.com/watch?v=m3bW8d4Brec)

# Plans for version 0.1
- Asset pipeline rework, I need something simplier, that allows user-generated content. 
- Asynchronous asset loading.
- Data-orientired scene-graph.
- Render graph.
- Some sort of framework based on specs (https://docs.rs/specs/latest/specs/)
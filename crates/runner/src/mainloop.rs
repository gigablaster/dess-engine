// Copyright (C) 2023 gigablaster

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use ash::vk;
use common::{Client, ClientState, TimeFilter};
use log::{log, Level};
use render::vulkan::{Device, Instance, PhysicalDeviceList, Surface, Swapchain};
use sdl2::{
    event::Event,
    log::{set_output_function, Category, Priority},
};

pub fn run(client: impl Client) -> Result<(), String> {
    let mut client = client;
    simple_logger::init().unwrap();
    let sdl = sdl2::init()?;
    set_output_function(sdl_log);
    let video = sdl.video()?;
    let timer = sdl.timer()?;

    let window = video
        .window("SexFarm", 1280, 720)
        .position_centered()
        .vulkan()
        .build()
        .map_err(|x| x.to_string())?;

    let mut event_pump = sdl.event_pump()?;

    let instance = Instance::builder()
        .debug(cfg!(debug_assertions))
        .build(&window)
        .unwrap();

    let surface = Surface::create(&instance, &window).unwrap();

    let pdevice = instance
        .enumerate_physical_devices()
        .unwrap()
        .find_suitable_device(
            &surface,
            &[
                vk::PhysicalDeviceType::DISCRETE_GPU,
                vk::PhysicalDeviceType::INTEGRATED_GPU,
            ],
        )
        .unwrap();

    let device = Device::create(&instance, &pdevice).unwrap();

    let _swapchain = Swapchain::new(&device, surface).unwrap();

    let result = {
        let mut time_filter = TimeFilter::new();
        let mut counter = timer.performance_counter();
        'running: loop {
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. } => break 'running,
                    _ => {}
                }
            }
            let new_counter = timer.performance_counter();
            let delta = new_counter - counter;
            counter = new_counter;
            let dt = delta as f64 / timer.performance_frequency() as f64;
            let game_time = time_filter.sample(dt);
            match client.tick(&game_time).unwrap() {
                ClientState::Continue => {
                    client.present(&game_time);
                }
                ClientState::Exit => break 'running,
            }
        }

        Ok(())
    };

    result
}

fn get_priority(priority: Priority) -> Level {
    match priority {
        Priority::Info => Level::Info,
        Priority::Warn => Level::Warn,
        Priority::Debug => Level::Debug,
        Priority::Error => Level::Error,
        Priority::Verbose => Level::Info,
        Priority::Critical => Level::Error,
    }
}

fn get_category(category: Category) -> &'static str {
    match category {
        Category::Test => "Test:",
        Category::Error => "Error:",
        Category::Audio => "Audio:",
        Category::Video => "Video:",
        Category::Input => "Input:",
        Category::Assert => "Assert:",
        Category::System => "System:",
        Category::Render => "Render:",
        Category::Custom => "",
        Category::Unknown => "",
        Category::Application => "",
    }
}

fn sdl_log(priority: Priority, category: Category, message: &str) {
    log!(
        target: "SDL",
        get_priority(priority),
        "{} {}",
        get_category(category),
        message
    );
}

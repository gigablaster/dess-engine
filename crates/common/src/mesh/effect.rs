use std::{collections::HashMap, io};

use crate::traits::{BinaryDeserialization, BinarySerialization};

#[derive(Debug)]
pub struct EffectInfo {
    pub name: String,
    pub textures: HashMap<String, String>,
    pub params: HashMap<String, glam::Vec4>,
}

impl EffectInfo {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            textures: HashMap::new(),
            params: HashMap::new(),
        }
    }

    pub fn add_texture(&mut self, slot: &str, texture: &str) {
        self.textures.insert(slot.into(), texture.into());
    }

    pub fn add_param(&mut self, slot: &str, value: glam::Vec4) {
        self.params.insert(slot.into(), value);
    }
}

impl BinaryDeserialization for EffectInfo {
    fn deserialize(r: &mut impl io::Read) -> io::Result<Self> {
        let name = String::deserialize(r)?;
        let textures = HashMap::deserialize(r)?;
        let params = HashMap::deserialize(r)?;

        Ok(Self {
            name,
            textures,
            params,
        })
    }
}

impl BinarySerialization for EffectInfo {
    fn serialize(&self, w: &mut impl io::Write) -> io::Result<()> {
        self.name.serialize(w)?;
        self.textures.serialize(w)?;
        self.params.serialize(w)?;

        Ok(())
    }
}

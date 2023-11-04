use std::{collections::HashMap, hash::Hash};

use byteorder::{ReadBytesExt, WriteBytesExt};
use dess_common::traits::{BinaryDeserialization, BinarySerialization};

use crate::Asset;

#[derive(Debug)]
pub enum GpuShaderStage {
    Vertex,
    Fragment,
}

#[derive(Debug)]
pub enum Error {
    DefinitionDontExist(String),
    ShaderVariationNotFound,
}

pub struct GpuShader {
    stage: GpuShaderStage,
    defines: Vec<String>,
    variations: HashMap<u32, Vec<u8>>,
}

impl GpuShader {
    pub fn new(stage: GpuShaderStage, possible_definitions: &[&str]) -> Self {
        assert!(possible_definitions.len() <= 32);
        Self {
            stage,
            defines: possible_definitions.iter().map(|x| x.to_string()).collect(),
            variations: HashMap::new(),
        }
    }

    pub fn get_shader_variant(&self, definitions: &[&str]) -> Result<&[u8], Error> {
        self.variations
            .get(&self.get_key(definitions)?)
            .map(|x| x.as_slice())
            .ok_or(Error::ShaderVariationNotFound)
    }

    pub fn add_shader_variant(
        &mut self,
        definitions: &[&str],
        bytecode: &[u8],
    ) -> Result<(), Error> {
        self.variations
            .insert(self.get_key(definitions)?, bytecode.into());

        Ok(())
    }

    fn get_key(&self, definitions: &[&str]) -> Result<u32, Error> {
        let mut key = 0;
        for definition in definitions.iter() {
            let index = self
                .get_definition_index(definition)
                .ok_or(Error::DefinitionDontExist(definition.to_string()))?;
            key |= 1 << index;
        }

        Ok(key)
    }

    fn get_definition_index(&self, definition: &str) -> Option<usize> {
        self.defines.iter().position(|x| x == definition)
    }
}

impl Asset for GpuShader {
    const TYPE_ID: uuid::Uuid = uuid::uuid!("05596e6e-01f2-4c42-97ad-6ea7d10a29e5");
    fn collect_dependencies(&self, _deps: &mut Vec<crate::AssetRef>) {}
}

const VERTEX_SHADER_ID: u8 = 0;
const FRAGMENT_SHADER_ID: u8 = 1;

impl BinarySerialization for GpuShader {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        match self.stage {
            GpuShaderStage::Vertex => w.write_u8(VERTEX_SHADER_ID),
            GpuShaderStage::Fragment => w.write_u8(FRAGMENT_SHADER_ID),
        }?;
        self.defines.serialize(w)?;
        self.variations.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for GpuShader {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let ty = r.read_u8()?;
        let stage = match ty {
            VERTEX_SHADER_ID => GpuShaderStage::Vertex,
            FRAGMENT_SHADER_ID => GpuShaderStage::Fragment,
            _ => panic!("Unknown shader stage {}", ty),
        };
        let defines = Vec::deserialize(r)?;
        let variations = HashMap::deserialize(r)?;

        Ok(Self {
            stage,
            defines,
            variations,
        })
    }
}

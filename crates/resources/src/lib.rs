use std::io;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use common::traits::{BinaryDeserialization, BinarySerialization};
use four_cc::FourCC;
use uuid::Uuid;

pub mod content;

const VERSION: u16 = 1;

const FOURCC: FourCC = FourCC(*b"dres");

struct SerializedResourceHeader {
    pub fourcc: FourCC,
    pub version: u16,
    pub type_id: Uuid,
}

impl SerializedResourceHeader {
    fn new(resource: &dyn SerializedResource) -> Self {
        Self {
            fourcc: FOURCC,
            version: VERSION,
            type_id: resource.type_id(),
        }
    }
}

impl BinarySerialization for SerializedResourceHeader {
    fn serialize(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        self.fourcc.serialize(w)?;
        w.write_u16::<LittleEndian>(self.version)?;
        self.type_id.serialize(w)?;

        Ok(())
    }
}

impl BinaryDeserialization for SerializedResourceHeader {
    fn deserialize(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let fourcc = FourCC::deserialize(r)?;
        let version = r.read_u16::<LittleEndian>()?;
        let type_id = Uuid::deserialize(r)?;

        Ok(Self {
            fourcc,
            version,
            type_id,
        })
    }
}

pub trait SerializedResource {
    fn type_id(&self) -> Uuid;
    fn serialize(&self, w: &mut dyn io::Write) -> io::Result<()>;
}

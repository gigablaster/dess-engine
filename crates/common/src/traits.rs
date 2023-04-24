use std::io::{self, Read, Write};

pub trait BinaryDeserialization: Sized {
    fn deserialize(r: &mut impl Read) -> io::Result<Self>;
}

pub trait BinarySerialization {
    fn serialize(&self, w: &mut impl Write) -> io::Result<()>;
}

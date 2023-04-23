use std::io::{self, Read, Write};

pub trait VfsRead: Sized {
    fn read(r: &mut impl Read) -> io::Result<Self>;
}

pub trait VfsWrite {
    fn write(&self, w: &mut impl Write) -> io::Result<()>;
}

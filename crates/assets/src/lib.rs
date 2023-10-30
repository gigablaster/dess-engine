use bytes::Bytes;
use turbosloth::Lazy;

mod image;

pub enum DataSource {
    Lazy(Lazy<Bytes>),
    Immediate(Bytes),
}

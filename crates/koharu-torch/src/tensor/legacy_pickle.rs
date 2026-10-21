//! Reader for PyTorch's legacy (pre-1.6, non-zip) `torch.save` format.
//!
//! libtorch only reads the zip container, so state dicts saved before PyTorch
//! 1.6 (many ESRGAN checkpoints) cannot go through `loadz_multi`. This follows
//! `torch/serialization.py` `_legacy_save`: five pickles (magic number,
//! protocol version, system info, the object, the sorted storage keys), then
//! each storage as a little-endian `i64` element count followed by its raw
//! data. Only what a state dict of plain tensors needs is supported; any other
//! opcode or callable fails closed.
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use crate::{Kind, TchError, Tensor};

const MAGIC_NUMBER: i128 = 0x1950a86a20f9469cfc6c;
const PROTOCOL_VERSION: i128 = 1001;

#[derive(Clone, Debug)]
enum Value {
    None,
    Bool,
    Int(i128),
    Str(String),
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage {
        key: String,
        kind: Kind,
    },
    Tensor {
        key: String,
        kind: Kind,
        offset: i64,
        size: Vec<i64>,
        stride: Vec<i64>,
    },
}

fn format_error(message: impl Into<String>) -> TchError {
    TchError::FileFormat(format!("legacy torch pickle: {}", message.into()))
}

struct Machine<'c, 'b> {
    input: &'c mut Cursor<&'b [u8]>,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<u32, Value>,
}

impl Machine<'_, '_> {
    fn byte(&mut self) -> Result<u8, TchError> {
        let mut buffer = [0; 1];
        self.input.read_exact(&mut buffer)?;
        Ok(buffer[0])
    }

    fn bytes(&mut self, count: usize) -> Result<Vec<u8>, TchError> {
        let mut buffer = vec![0; count];
        self.input.read_exact(&mut buffer)?;
        Ok(buffer)
    }

    fn u16(&mut self) -> Result<u16, TchError> {
        let bytes = self.bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, TchError> {
        let bytes = self.bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn line(&mut self) -> Result<String, TchError> {
        let mut line = Vec::new();
        loop {
            match self.byte()? {
                b'\n' => break,
                byte => line.push(byte),
            }
        }
        String::from_utf8(line).map_err(|_| format_error("GLOBAL name is not UTF-8"))
    }

    fn pop(&mut self) -> Result<Value, TchError> {
        self.stack
            .pop()
            .ok_or_else(|| format_error("stack underflow"))
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>, TchError> {
        let mark = self
            .marks
            .pop()
            .ok_or_else(|| format_error("missing MARK"))?;
        if mark > self.stack.len() {
            return Err(format_error("MARK beyond the stack"));
        }
        Ok(self.stack.split_off(mark))
    }

    fn top(&mut self) -> Result<&mut Value, TchError> {
        self.stack
            .last_mut()
            .ok_or_else(|| format_error("stack underflow"))
    }

    fn string(&mut self, length: usize) -> Result<(), TchError> {
        let bytes = self.bytes(length)?;
        let value = String::from_utf8(bytes).map_err(|_| format_error("string is not UTF-8"))?;
        self.stack.push(Value::Str(value));
        Ok(())
    }

    fn run(mut self) -> Result<Value, TchError> {
        loop {
            match self.byte()? {
                0x80 => {
                    // PROTO
                    self.byte()?;
                }
                0x95 => {
                    // FRAME
                    self.bytes(8)?;
                }
                b'.' => return self.pop(),
                b'(' => self.marks.push(self.stack.len()),
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool),
                0x89 => self.stack.push(Value::Bool),
                b'K' => {
                    let value = self.byte()?;
                    self.stack.push(Value::Int(i128::from(value)));
                }
                b'M' => {
                    let value = self.u16()?;
                    self.stack.push(Value::Int(i128::from(value)));
                }
                b'J' => {
                    let value = self.u32()? as i32;
                    self.stack.push(Value::Int(i128::from(value)));
                }
                0x8a => {
                    // LONG1: little-endian two's complement.
                    let length = usize::from(self.byte()?);
                    if length > 16 {
                        return Err(format_error("LONG1 is too long"));
                    }
                    let bytes = self.bytes(length)?;
                    let mut value: i128 = 0;
                    for (index, byte) in bytes.iter().enumerate() {
                        value |= i128::from(*byte) << (8 * index);
                    }
                    if (1..16).contains(&length) && bytes[length - 1] & 0x80 != 0 {
                        value -= 1i128 << (8 * length);
                    }
                    self.stack.push(Value::Int(value));
                }
                b'X' => {
                    let length = self.u32()? as usize;
                    self.string(length)?;
                }
                0x8c => {
                    let length = usize::from(self.byte()?);
                    self.string(length)?;
                }
                b'c' => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.stack.push(Value::Global(module, name));
                }
                b'q' => {
                    let index = u32::from(self.byte()?);
                    let value = self.top()?.clone();
                    self.memo.insert(index, value);
                }
                b'r' => {
                    let index = self.u32()?;
                    let value = self.top()?.clone();
                    self.memo.insert(index, value);
                }
                0x94 => {
                    // MEMOIZE
                    let index = self.memo.len() as u32;
                    let value = self.top()?.clone();
                    self.memo.insert(index, value);
                }
                b'h' => {
                    let index = u32::from(self.byte()?);
                    self.get(index)?;
                }
                b'j' => {
                    let index = self.u32()?;
                    self.get(index)?;
                }
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b']' => self.stack.push(Value::List(Vec::new())),
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85 => {
                    let first = self.pop()?;
                    self.stack.push(Value::Tuple(vec![first]));
                }
                0x86 => {
                    let second = self.pop()?;
                    let first = self.pop()?;
                    self.stack.push(Value::Tuple(vec![first, second]));
                }
                0x87 => {
                    let third = self.pop()?;
                    let second = self.pop()?;
                    let first = self.pop()?;
                    self.stack.push(Value::Tuple(vec![first, second, third]));
                }
                b'a' => {
                    let item = self.pop()?;
                    match self.top()? {
                        Value::List(items) => items.push(item),
                        _ => return Err(format_error("APPEND target is not a list")),
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    match self.top()? {
                        Value::List(list) => list.extend(items),
                        _ => return Err(format_error("APPENDS target is not a list")),
                    }
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    match self.top()? {
                        Value::Dict(entries) => entries.push((key, value)),
                        _ => return Err(format_error("SETITEM target is not a dict")),
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(format_error("SETITEMS has an odd item count"));
                    }
                    let mut pairs = Vec::with_capacity(items.len() / 2);
                    let mut items = items.into_iter();
                    while let (Some(key), Some(value)) = (items.next(), items.next()) {
                        pairs.push((key, value));
                    }
                    match self.top()? {
                        Value::Dict(entries) => entries.extend(pairs),
                        _ => return Err(format_error("SETITEMS target is not a dict")),
                    }
                }
                b'Q' => {
                    let id = self.pop()?;
                    let storage = persistent_storage(id)?;
                    self.stack.push(storage);
                }
                b'R' => {
                    let arguments = self.pop()?;
                    let callable = self.pop()?;
                    let value = reduce(callable, arguments)?;
                    self.stack.push(value);
                }
                b'b' => {
                    // BUILD: an OrderedDict's `_metadata` attribute carries only
                    // module versions; a state dict does not need it.
                    let _state = self.pop()?;
                    if !matches!(self.top()?, Value::Dict(_)) {
                        return Err(format_error("BUILD target is not a dict"));
                    }
                }
                opcode => {
                    return Err(format_error(format!("unsupported opcode 0x{opcode:02x}")));
                }
            }
        }
    }

    fn get(&mut self, index: u32) -> Result<(), TchError> {
        let value = self
            .memo
            .get(&index)
            .cloned()
            .ok_or_else(|| format_error(format!("memo {index} is missing")))?;
        self.stack.push(value);
        Ok(())
    }
}

fn storage_kind(module: &str, name: &str) -> Result<Kind, TchError> {
    match (module, name) {
        ("torch", "FloatStorage") => Ok(Kind::Float),
        ("torch", "HalfStorage") => Ok(Kind::Half),
        ("torch", "DoubleStorage") => Ok(Kind::Double),
        _ => Err(format_error(format!("unsupported storage {module}.{name}"))),
    }
}

fn persistent_storage(id: Value) -> Result<Value, TchError> {
    let Value::Tuple(items) = id else {
        return Err(format_error("persistent id is not a tuple"));
    };
    match items.as_slice() {
        [Value::Str(tag), Value::Global(module, name), Value::Str(key), Value::Str(_location), Value::Int(_numel), Value::None]
            if tag == "storage" =>
        {
            Ok(Value::Storage {
                key: key.clone(),
                kind: storage_kind(module, name)?,
            })
        }
        _ => Err(format_error("persistent id is not a plain storage")),
    }
}

fn ints(value: &Value) -> Result<Vec<i64>, TchError> {
    match value {
        Value::Tuple(items) => items
            .iter()
            .map(|item| match item {
                Value::Int(value) => {
                    i64::try_from(*value).map_err(|_| format_error("tensor shape overflows"))
                }
                _ => Err(format_error("tensor shape is not integers")),
            })
            .collect(),
        _ => Err(format_error("tensor shape is not a tuple")),
    }
}

fn reduce(callable: Value, arguments: Value) -> Result<Value, TchError> {
    let Value::Global(module, name) = callable else {
        return Err(format_error("REDUCE callable is not a global"));
    };
    let Value::Tuple(arguments) = arguments else {
        return Err(format_error("REDUCE arguments are not a tuple"));
    };
    match (module.as_str(), name.as_str(), arguments.as_slice()) {
        ("collections", "OrderedDict", []) => Ok(Value::Dict(Vec::new())),
        (
            "torch._utils",
            "_rebuild_tensor_v2",
            [Value::Storage { key, kind }, Value::Int(offset), size, stride, ..],
        ) => Ok(Value::Tensor {
            key: key.clone(),
            kind: *kind,
            offset: i64::try_from(*offset).map_err(|_| format_error("storage offset overflows"))?,
            size: ints(size)?,
            stride: ints(stride)?,
        }),
        _ => Err(format_error(format!(
            "unsupported callable {module}.{name}"
        ))),
    }
}

fn unpickle(input: &mut Cursor<&[u8]>) -> Result<Value, TchError> {
    Machine {
        input,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: HashMap::new(),
    }
    .run()
}

/// Whether `path` holds a legacy `torch.save` file rather than a zip.
pub(crate) fn is_legacy(path: &Path) -> Result<bool, TchError> {
    let mut magic = [0; 2];
    std::fs::File::open(path)?.read_exact(&mut magic)?;
    Ok(magic[0] == 0x80)
}

/// Reads the named tensors of a legacy state dict.
pub(crate) fn read<T: AsRef<Path>>(path: T) -> Result<Vec<(String, Tensor)>, TchError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    read_bytes(&bytes).map_err(|error| match error {
        TchError::FileFormat(message) => {
            TchError::FileFormat(format!("{message} in {}", path.display()))
        }
        error => error,
    })
}

fn read_bytes(bytes: &[u8]) -> Result<Vec<(String, Tensor)>, TchError> {
    let mut cursor = Cursor::new(bytes);
    match unpickle(&mut cursor)? {
        Value::Int(MAGIC_NUMBER) => {}
        _ => return Err(format_error("magic number does not match")),
    }
    match unpickle(&mut cursor)? {
        Value::Int(PROTOCOL_VERSION) => {}
        _ => return Err(format_error("protocol version is not 1001")),
    }
    unpickle(&mut cursor)?;
    let Value::Dict(entries) = unpickle(&mut cursor)? else {
        return Err(format_error("the saved object is not a state dict"));
    };
    let Value::List(keys) = unpickle(&mut cursor)? else {
        return Err(format_error("storage keys are not a list"));
    };

    // A storage's element size is known only from the tensors that view it.
    let mut kinds = HashMap::new();
    for (_, value) in &entries {
        if let Value::Tensor { key, kind, .. } = value {
            if kinds
                .insert(key.clone(), *kind)
                .is_some_and(|other| other != *kind)
            {
                return Err(format_error(format!("storage {key} has two element kinds")));
            }
        }
    }
    let mut offset = cursor.position() as usize;
    let mut storages = HashMap::new();
    for key in keys {
        let Value::Str(key) = key else {
            return Err(format_error("storage key is not a string"));
        };
        let header = bytes
            .get(offset..offset + 8)
            .ok_or_else(|| format_error("storage header is truncated"))?;
        let count = i64::from_le_bytes(header.try_into().expect("eight bytes"));
        offset += 8;
        let kind = *kinds
            .get(&key)
            .ok_or_else(|| format_error(format!("storage {key} is not used by any tensor")))?;
        let end = usize::try_from(count)
            .ok()
            .and_then(|count| count.checked_mul(kind.elt_size_in_bytes()))
            .and_then(|length| offset.checked_add(length))
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| format_error(format!("storage {key} is truncated")))?;
        storages.insert(key, (offset..end, count));
        offset = end;
    }

    entries
        .into_iter()
        .map(|(name, value)| {
            let Value::Str(name) = name else {
                return Err(format_error("state dict key is not a string"));
            };
            let Value::Tensor {
                key,
                kind,
                offset,
                size,
                stride,
            } = value
            else {
                return Err(format_error(format!("{name} is not a tensor")));
            };
            let (range, count) = storages
                .get(&key)
                .ok_or_else(|| format_error(format!("storage {key} of {name} is missing")))?;
            let storage = Tensor::f_from_data_size(&bytes[range.clone()], &[*count], kind)?;
            let tensor = storage
                .f_as_strided(&size[..], &stride[..], offset)?
                .f_contiguous()?;
            Ok((name, tensor))
        })
        .collect()
}

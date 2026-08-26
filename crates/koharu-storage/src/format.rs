use std::{
    fs,
    path::{Path, PathBuf},
};

use revision::revisioned;
use serde::{Deserialize, Serialize};

use crate::{BlobId, DocumentId, Error, Result, Revision, VersionId, durability};

const MAGIC: &[u8; 8] = b"KHRSTATE";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 8 + 4 + 8 + 32;
const MAX_STATE_BYTES: usize = 512 * 1024 * 1024;
const VERSION_MAGIC: &[u8; 8] = b"KHRVERSN";
const VERSIONS_DIRECTORY: &str = "versions";

#[revisioned(revision = 1)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct StoredState {
    pub(crate) document: DocumentId,
    pub(crate) revision: Revision,
    pub(crate) blobs: Vec<BlobId>,
    pub(crate) payload: Vec<u8>,
}

#[revisioned(revision = 1)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct StoredVersion {
    pub(crate) id: VersionId,
    pub(crate) name: String,
    pub(crate) created_at_ms: u64,
    pub(crate) state: StoredState,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum Slot {
    A,
    B,
}

impl Slot {
    pub(crate) const fn other(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }

    fn filename(self) -> &'static str {
        match self {
            Self::A => "state-a.khr",
            Self::B => "state-b.khr",
        }
    }
}

pub(crate) fn path(root: &Path, slot: Slot) -> PathBuf {
    root.join(slot.filename())
}

pub(crate) fn load(root: &Path, slot: Slot) -> Result<Option<StoredState>> {
    let path = path(root, slot);
    if !path.try_exists()? {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    decode(&bytes).map(Some)
}

pub(crate) fn save(root: &Path, slot: Slot, state: &StoredState) -> Result<()> {
    let bytes = encode(state)?;
    durability::publish(&path(root, slot), &bytes)
}

pub(crate) fn list_versions(root: &Path) -> Result<Vec<StoredVersion>> {
    let directory = root.join(VERSIONS_DIRECTORY);
    if !directory.try_exists()? {
        return Ok(Vec::new());
    }
    let mut versions = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if !entry.file_type()?.is_file()
            || entry.path().extension().and_then(|value| value.to_str()) != Some("khr")
        {
            continue;
        }
        let id = entry
            .path()
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| Error::invalid("project version has an invalid filename"))?
            .parse::<VersionId>()
            .map_err(|_| Error::invalid("project version filename is not a UUID"))?;
        let version = decode_version(&fs::read(entry.path())?)?;
        if version.id != id {
            return Err(Error::invalid(
                "project version filename and ID do not match",
            ));
        }
        versions.push(version);
    }
    versions.sort_unstable_by(|left, right| {
        right
            .created_at_ms
            .cmp(&left.created_at_ms)
            .then_with(|| right.id.cmp(&left.id))
    });
    Ok(versions)
}

pub(crate) fn load_version(root: &Path, id: VersionId) -> Result<Option<StoredVersion>> {
    let path = version_path(root, id);
    if !path.try_exists()? {
        return Ok(None);
    }
    let version = decode_version(&fs::read(path)?)?;
    if version.id != id {
        return Err(Error::invalid(
            "project version filename and ID do not match",
        ));
    }
    Ok(Some(version))
}

pub(crate) fn save_version(root: &Path, version: &StoredVersion) -> Result<()> {
    durability::publish(&version_path(root, version.id), &encode_version(version)?)
}

pub(crate) fn delete_version(root: &Path, id: VersionId) -> Result<bool> {
    durability::remove(&version_path(root, id))
}

fn version_path(root: &Path, id: VersionId) -> PathBuf {
    root.join(VERSIONS_DIRECTORY).join(format!("{id}.khr"))
}

fn encode(state: &StoredState) -> Result<Vec<u8>> {
    validate(state)?;
    let body = revision::to_vec(state)?;
    if body.len() > MAX_STATE_BYTES {
        return Err(Error::invalid("state exceeds the maximum encoded size"));
    }
    let checksum = blake3::hash(&body);
    let mut encoded = Vec::with_capacity(HEADER_BYTES + body.len());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&VERSION.to_le_bytes());
    encoded.extend_from_slice(&(body.len() as u64).to_le_bytes());
    encoded.extend_from_slice(checksum.as_bytes());
    encoded.extend_from_slice(&body);
    Ok(encoded)
}

fn decode(encoded: &[u8]) -> Result<StoredState> {
    if encoded.len() < HEADER_BYTES || &encoded[..8] != MAGIC {
        return Err(Error::NotAProject);
    }
    let version = u32::from_le_bytes(encoded[8..12].try_into().expect("fixed header range"));
    if version != VERSION {
        return Err(Error::UnsupportedFormat(version));
    }
    let body_len = u64::from_le_bytes(encoded[12..20].try_into().expect("fixed header range"));
    let body_len = usize::try_from(body_len)
        .map_err(|_| Error::invalid("state length does not fit this platform"))?;
    if body_len > MAX_STATE_BYTES || encoded.len() != HEADER_BYTES + body_len {
        return Err(Error::invalid("state length is invalid"));
    }
    let expected = &encoded[20..52];
    let body = &encoded[HEADER_BYTES..];
    if blake3::hash(body).as_bytes() != expected {
        return Err(Error::invalid("state checksum mismatch"));
    }
    let state: StoredState = revision::from_slice(body)?;
    validate(&state)?;
    Ok(state)
}

fn validate(state: &StoredState) -> Result<()> {
    if state.payload.len() > MAX_STATE_BYTES {
        return Err(Error::invalid("scene payload exceeds the maximum size"));
    }
    if state.blobs.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Error::invalid("state blob IDs are not sorted and unique"));
    }
    Ok(())
}

fn encode_version(version: &StoredVersion) -> Result<Vec<u8>> {
    validate(&version.state)?;
    let body = revision::to_vec(version)?;
    if body.len() > MAX_STATE_BYTES {
        return Err(Error::invalid(
            "project version exceeds the maximum encoded size",
        ));
    }
    let checksum = blake3::hash(&body);
    let mut encoded = Vec::with_capacity(HEADER_BYTES + body.len());
    encoded.extend_from_slice(VERSION_MAGIC);
    encoded.extend_from_slice(&VERSION.to_le_bytes());
    encoded.extend_from_slice(&(body.len() as u64).to_le_bytes());
    encoded.extend_from_slice(checksum.as_bytes());
    encoded.extend_from_slice(&body);
    Ok(encoded)
}

fn decode_version(encoded: &[u8]) -> Result<StoredVersion> {
    if encoded.len() < HEADER_BYTES || &encoded[..8] != VERSION_MAGIC {
        return Err(Error::invalid("project version header is invalid"));
    }
    let version = u32::from_le_bytes(encoded[8..12].try_into().expect("fixed header range"));
    if version != VERSION {
        return Err(Error::UnsupportedFormat(version));
    }
    let body_len = u64::from_le_bytes(encoded[12..20].try_into().expect("fixed header range"));
    let body_len = usize::try_from(body_len)
        .map_err(|_| Error::invalid("project version length does not fit this platform"))?;
    if body_len > MAX_STATE_BYTES || encoded.len() != HEADER_BYTES + body_len {
        return Err(Error::invalid("project version length is invalid"));
    }
    let expected = &encoded[20..52];
    let body = &encoded[HEADER_BYTES..];
    if blake3::hash(body).as_bytes() != expected {
        return Err(Error::invalid("project version checksum mismatch"));
    }
    let version: StoredVersion = revision::from_slice(body)?;
    validate(&version.state)?;
    Ok(version)
}

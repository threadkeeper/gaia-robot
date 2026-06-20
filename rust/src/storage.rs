#![allow(dead_code)]

//! Knowledge-base and data-lake record models for Gaia.
//!
//! The design mirrors the Cosmos DB layout described in `infra/README.md`:
//! - KB records are user- and entity-centric knowledge snapshots.
//! - DL records are larger, append-oriented data-lake documents.
//! - Each record is keyed by a stable `record_id` and carries the business
//!   key (`user_id` or `entity_id`), a `date`, and a `data_vector` for
//!   similarity search.
//!
//! The serde field names deliberately match the Cosmos document shape produced
//! by `infra/cosmos_create.py` / `migrations/loadData.py`, so a [`Record`] can
//! be deserialized straight from a Cosmos query result and serialized back for
//! an upsert: `record_id` <-> `id`, `entity_id` <-> `entity`, `user_id` <->
//! `userId`, and `data_vector` <-> `dataVector` (the embedding path indexed by
//! DiskANN).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A storage record that can be written with an upsert-style operation.
///
/// The field renames map this domain type onto the on-disk Cosmos document, so
/// the same struct is used both in-memory and on the wire. Unknown Cosmos fields
/// (e.g. `wing`, `source`, `items`, `_migration`) are ignored on deserialize.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Stable Cosmos document id (the system `/id` field).
    #[serde(rename = "id")]
    pub record_id: String,
    /// Entity partition key for the `Gaia*` containers (`/entity`). Empty for
    /// user-partitioned containers.
    #[serde(rename = "entity", default, skip_serializing_if = "String::is_empty")]
    pub entity_id: String,
    /// User partition key for the `Users*` containers (`/userId`). Empty for
    /// entity-partitioned containers.
    #[serde(rename = "userId", default, skip_serializing_if = "String::is_empty")]
    pub user_id: String,
    /// The record day (`/date`); unique within a logical partition. May be
    /// absent in some containers (e.g. GaiaDiary entries without a date).
    #[serde(default)]
    pub date: String,
    /// Logical family (KB vs DL). Derived from the container, not stored, so it
    /// is skipped during (de)serialization.
    #[serde(skip)]
    pub kind: RecordKind,
    /// The record's text payload; its embedding lives at `/dataVector`.
    /// Defaults to empty when absent (some containers omit it).
    #[serde(default)]
    pub data: String,
    /// The embedding of `data`, stored and DiskANN-indexed at Cosmos path
    /// `/dataVector`. Empty until an embedding has been computed.
    #[serde(rename = "dataVector", default, skip_serializing_if = "Vec::is_empty")]
    pub data_vector: Vec<f32>,
    /// Optional free-form metadata; omitted from the document when empty.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

/// Describes the index shape that a table should expose for retrieval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSpec {
    /// Stable name used by the application or the storage layer.
    pub name: &'static str,
    /// Fields that participate in this index.
    pub fields: Vec<&'static str>,
    /// True when the index supports vector similarity lookups.
    pub supports_vector_search: bool,
    /// True when the index supports date-range filters.
    pub supports_date_range: bool,
}

/// Describes the logical layout of a KB or DL container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableSchema {
    /// Container name, such as `GaiaKB` or `UsersDataLake`.
    pub name: &'static str,
    /// The logical family: knowledge base or data lake.
    pub kind: RecordKind,
    /// The partition key used to filter by user or entity.
    pub partition_key: &'static str,
    /// The indexes that should exist for retrieval and filtering.
    pub indexes: Vec<IndexSpec>,
}

impl TableSchema {
    /// Build the standard KB/DL index set expected by Gaia.
    pub fn standard(kind: RecordKind, name: &'static str, partition_key: &'static str) -> Self {
        Self {
            name,
            kind,
            partition_key,
            indexes: vec![
                IndexSpec {
                    name: "user_entity_filter",
                    fields: vec![partition_key],
                    supports_vector_search: false,
                    supports_date_range: false,
                },
                IndexSpec {
                    name: "date_filter",
                    fields: vec!["date"],
                    supports_vector_search: false,
                    supports_date_range: true,
                },
                IndexSpec {
                    name: "data_vector_index",
                    fields: vec!["dataVector"],
                    supports_vector_search: true,
                    supports_date_range: false,
                },
            ],
        }
    }
}

/// The logical record family that determines the container / index set.
///
/// Defaults to [`RecordKind::KnowledgeBase`] so it can be the `#[serde(skip)]`
/// default when a [`Record`] is deserialized from a Cosmos document (where the
/// family is implied by the container rather than stored on the document).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordKind {
    #[default]
    KnowledgeBase,
    DataLake,
}

impl Record {
    /// Create a canonical record for a KB or DL write.
    pub fn new(
        record_id: impl Into<String>,
        entity_id: impl Into<String>,
        user_id: impl Into<String>,
        date: impl Into<String>,
        kind: RecordKind,
        data: impl Into<String>,
        data_vector: Vec<f32>,
    ) -> Self {
        Self {
            record_id: record_id.into(),
            entity_id: entity_id.into(),
            user_id: user_id.into(),
            date: date.into(),
            kind,
            data: data.into(),
            data_vector,
            metadata: BTreeMap::new(),
        }
    }

    /// Return the business partition key used for filtering by user or entity.
    pub fn business_key(&self) -> &str {
        if self.user_id.is_empty() {
            &self.entity_id
        } else {
            &self.user_id
        }
    }

    /// Return the embedding (`/dataVector`) values for indexing and retrieval.
    pub fn data_vector_bytes(&self) -> &[f32] {
        &self.data_vector
    }
}

/// A tiny in-memory repository that models the expected UPSERT behavior.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct Repository {
    records: BTreeMap<String, Record>,
}

/// A KB-oriented write facade that uses the standard schema and UPSERT semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeBaseTable {
    schema: TableSchema,
}

impl KnowledgeBaseTable {
    /// Create a KB table with the standard Gaia indexes.
    pub fn new(name: &'static str, partition_key: &'static str) -> Self {
        Self {
            schema: TableSchema::standard(RecordKind::KnowledgeBase, name, partition_key),
        }
    }

    /// Upsert one KB record into the repository.
    pub fn upsert(&self, repo: &mut Repository, record: Record) {
        debug_assert_eq!(record.kind, RecordKind::KnowledgeBase);
        repo.upsert(record);
    }

    /// Return the table schema that describes the KB indexes.
    pub fn schema(&self) -> &TableSchema {
        &self.schema
    }
}

/// A DL-oriented write facade that uses the standard schema and UPSERT semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataLakeTable {
    schema: TableSchema,
}

impl DataLakeTable {
    /// Create a DL table with the standard Gaia indexes.
    pub fn new(name: &'static str, partition_key: &'static str) -> Self {
        Self {
            schema: TableSchema::standard(RecordKind::DataLake, name, partition_key),
        }
    }

    /// Upsert one DL record into the repository.
    pub fn upsert(&self, repo: &mut Repository, record: Record) {
        debug_assert_eq!(record.kind, RecordKind::DataLake);
        repo.upsert(record);
    }

    /// Return the table schema that describes the DL indexes.
    pub fn schema(&self) -> &TableSchema {
        &self.schema
    }
}

impl Repository {
    /// Upsert one record into the repository.
    pub fn upsert(&mut self, record: Record) {
        self.records.insert(record.record_id.clone(), record);
    }

    /// Find a record by its stable id.
    pub fn get(&self, record_id: &str) -> Option<&Record> {
        self.records.get(record_id)
    }

    /// Return the stored records filtered by the business key and date range.
    pub fn query(
        &self,
        business_key: &str,
        from_date: Option<&str>,
        to_date: Option<&str>,
    ) -> Vec<&Record> {
        self.records
            .values()
            .filter(|record| record.business_key() == business_key)
            .filter(|record| from_date.map_or(true, |from| record.date.as_str() >= from))
            .filter(|record| to_date.map_or(true, |to| record.date.as_str() <= to))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upsert_replaces_an_existing_record_with_the_same_id() {
        let mut repo = Repository::default();

        let first = Record::new(
            "r-1",
            "entity-1",
            "user-1",
            "2026-06-16",
            RecordKind::KnowledgeBase,
            "alpha",
            vec![0.1, 0.2],
        );

        repo.upsert(first.clone());
        repo.upsert(Record::new(
            "r-1",
            "entity-1",
            "user-1",
            "2026-06-16",
            RecordKind::KnowledgeBase,
            "alpha-updated",
            vec![0.3, 0.4],
        ));

        assert_eq!(repo.get("r-1").unwrap().data, "alpha-updated");
        assert_eq!(repo.get("r-1").unwrap().data_vector, vec![0.3, 0.4]);
    }

    #[test]
    fn kb_and_dl_tables_expose_standard_indexes() {
        let kb = KnowledgeBaseTable::new("GaiaKB", "entity");
        let dl = DataLakeTable::new("UsersDataLake", "userId");

        assert_eq!(kb.schema().kind, RecordKind::KnowledgeBase);
        assert_eq!(dl.schema().kind, RecordKind::DataLake);
        assert!(kb
            .schema()
            .indexes
            .iter()
            .any(|index| index.name == "date_filter" && index.supports_date_range));
        assert!(dl
            .schema()
            .indexes
            .iter()
            .any(|index| index.name == "data_vector_index" && index.supports_vector_search));
        assert!(kb
            .schema()
            .indexes
            .iter()
            .any(|index| index.fields.contains(&"entity")));
        assert!(dl
            .schema()
            .indexes
            .iter()
            .any(|index| index.fields.contains(&"userId")));
    }

    #[test]
    fn query_filters_by_business_key_and_date_range() {
        let mut repo = Repository::default();
        repo.upsert(Record::new(
            "r-1",
            "entity-1",
            "user-1",
            "2026-06-15",
            RecordKind::KnowledgeBase,
            "one",
            vec![],
        ));
        repo.upsert(Record::new(
            "r-2",
            "entity-1",
            "user-1",
            "2026-06-16",
            RecordKind::DataLake,
            "two",
            vec![],
        ));
        repo.upsert(Record::new(
            "r-3",
            "entity-2",
            "user-2",
            "2026-06-16",
            RecordKind::KnowledgeBase,
            "three",
            vec![],
        ));

        let matches = repo.query("user-1", Some("2026-06-16"), Some("2026-06-16"));

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].record_id, "r-2");
    }

    #[test]
    fn record_serializes_to_the_cosmos_document_shape() {
        // A user-partitioned record should map onto the on-disk field names and
        // omit the empty/derived fields (entity, kind, metadata).
        let record = Record::new(
            "UsersDataLake_user-1_2026-06-16",
            "",
            "user-1",
            "2026-06-16",
            RecordKind::DataLake,
            "hello",
            vec![0.5, 0.25],
        );

        let value = serde_json::to_value(&record).unwrap();

        assert_eq!(value["id"], "UsersDataLake_user-1_2026-06-16");
        assert_eq!(value["userId"], "user-1");
        assert_eq!(value["date"], "2026-06-16");
        assert_eq!(value["data"], "hello");
        assert_eq!(value["dataVector"], serde_json::json!([0.5, 0.25]));
        // Empty entity, derived kind, and empty metadata are not written.
        assert!(value.get("entity").is_none());
        assert!(value.get("kind").is_none());
        assert!(value.get("metadata").is_none());
    }

    #[test]
    fn record_deserializes_from_a_cosmos_document_ignoring_extra_fields() {
        // Mirrors a document produced by migrations/loadData.py: it carries the
        // business key, date, data and extra fields the domain model ignores.
        let raw = r#"{
            "id": "GaiaKB_rust_2026-06-16",
            "entity": "rust",
            "date": "2026-06-16",
            "data": "facts about rust",
            "dataVector": [0.1, 0.2, 0.3],
            "wing": "gaia",
            "source": "kb",
            "count": 2,
            "items": [{"text": "a"}, {"text": "b"}],
            "_migration": true
        }"#;

        let record: Record = serde_json::from_str(raw).unwrap();

        assert_eq!(record.record_id, "GaiaKB_rust_2026-06-16");
        assert_eq!(record.entity_id, "rust");
        assert_eq!(record.user_id, "");
        assert_eq!(record.business_key(), "rust");
        assert_eq!(record.date, "2026-06-16");
        assert_eq!(record.data, "facts about rust");
        assert_eq!(record.data_vector, vec![0.1, 0.2, 0.3]);
        // `kind` is not stored, so it falls back to the default.
        assert_eq!(record.kind, RecordKind::default());
    }
}

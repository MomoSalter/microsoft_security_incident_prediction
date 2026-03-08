import pandas as pd
import json

INCIDENT_KEY     = 'IncidentId'
TARGET_COL       = 'IncidentGrade'

ENTITY_ID_COLS   = [
    'DetectorId', 'AlertTitle', 'DeviceId', 'Sha256', 'IpAddress', 'Url',
    'AccountSid', 'AccountUpn', 'AccountObjectId', 'AccountName', 'DeviceName',
    'NetworkMessageId', 'RegistryKey', 'RegistryValueName', 'RegistryValueData',
    'ApplicationId', 'ApplicationName', 'OAuthApplicationId', 'FileName',
    'FolderPath', 'ResourceIdName', 'OSFamily', 'OSVersion',
    'CountryCode', 'State', 'City', 'AlertId', "EmailClusterId"
]

CATEGORICAL_COLS = [
    'Category', 'MitreTechniques', 'EntityType',
    'EvidenceRole', 'SuspicionLevel', 'LastVerdict','ThreatFamily',
    'ResourceType', 'AntispamDirection', 'Roles',               
]

def categorical_feature_mapping(data):
    mappings = {}
    for col in CATEGORICAL_COLS:
        counts = data[col].value_counts()
        kept_values = counts[counts >= 1500].index.tolist()
        mappings[col] = kept_values
    return mappings

def organization_stats(data):
    org_grade = (
        data.groupby(['OrgId', TARGET_COL])
        .size()
        .unstack(fill_value=0)
    )

    org_grade = org_grade.div(org_grade.sum(axis=1), axis=0)
    org_grade.columns = [f'org_rate_{c}' for c in org_grade.columns]
    org_grade = org_grade.reset_index()

    # total incident count per org
    org_size = data.groupby('OrgId')[INCIDENT_KEY].nunique().rename('org_incident_count')
    org_stats = org_grade.merge(org_size, on='OrgId')
    return org_stats

def aggregate_incidents(df, include_target=True):
    with open("artifacts/cat_mappings.json") as f:
        mappings = json.load(f)
    org_stats = pd.read_parquet("artifacts/org_stats.parquet")

    result = []
    grouped = df.groupby(INCIDENT_KEY)

    if include_target:
        target = grouped[TARGET_COL].first().rename(TARGET_COL)
        result.append(target)
    
    evidence_count = grouped.size().rename("evidence_count")
    result.append(evidence_count)

    id_columns_agg = (grouped[ENTITY_ID_COLS].nunique().add_suffix("_nunique"))
    result.append(id_columns_agg)

    ts = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
    df_ts = df.copy()
    df_ts['_ts'] = ts

    ts_agg = df_ts.groupby(INCIDENT_KEY)['_ts'].agg(
        first_seen='min',
        last_seen='max'
    )
    ts_agg['duration_seconds'] = (
        ts_agg['last_seen'] - ts_agg['first_seen']
    ).dt.total_seconds().fillna(0).clip(lower=0)
    ts_agg['hour_of_day']  = ts_agg['first_seen'].dt.hour.fillna(-1)
    ts_agg['day_of_week']  = ts_agg['first_seen'].dt.dayofweek.fillna(-1)
    result.append(ts_agg[['duration_seconds', 'hour_of_day', 'day_of_week']])

    org_per_incident = grouped['OrgId'].first().rename('OrgId')
    result.append(org_per_incident)


    for column, values in mappings.items():
        new_columns = df[column].where(df[column].isin(values) | df[column].isna(), other="Other")
        tab = pd.crosstab(index=df[INCIDENT_KEY], columns=new_columns)
        tab.columns = [f'{column}_{v}' for v in tab.columns]
        result.append(tab)
    result = pd.concat(result, axis=1).reset_index()

    if include_target:
        result = result.dropna(subset=[TARGET_COL])

    result = result.fillna(0)

    result = result.merge(org_stats, on='OrgId', how='left')
    org_rate_cols = [c for c in org_stats.columns if c != 'OrgId']
    result[org_rate_cols] = result[org_rate_cols].fillna(org_stats[org_rate_cols].mean().to_dict())
    result = result.drop(columns=['OrgId'])

    return result
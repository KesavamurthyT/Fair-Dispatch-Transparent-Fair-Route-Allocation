# Fair Dispatch System API Integration Guide

This document describes how external delivery services can integrate with the Fair Dispatch System allocation API.

## Overview

The Fair Dispatch System provides a **single seamless API** for fair route allocation. External systems send drivers and packages as JSON, and receive optimized assignments with fairness metrics.

- **Endpoint**: `POST /api/v1/allocate`
- **Content-Type**: `application/json`
- **Authentication**: None required for demo (configure for production)

---

## Request Schema: AllocationRequest

```json
{
  "allocation_date": "2026-02-10",
  "warehouse": {
    "lat": 12.9716,
    "lng": 77.5946
  },
  "drivers": [
    {
      "id": "external_driver_001",
      "name": "Raju",
      "preferred_language": "en",
      "vehicle_capacity_kg": 150
    }
  ],
  "packages": [
    {
      "id": "external_pkg_001",
      "weight_kg": 2.5,
      "fragility_level": 3,
      "address": "Some street, Area, City",
      "latitude": 12.97,
      "longitude": 77.6,
      "priority": "NORMAL"
    }
  ]
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `allocation_date` | string (date) | Yes | Date for the allocation (YYYY-MM-DD) |
| `warehouse.lat` | float | Yes | Warehouse latitude |
| `warehouse.lng` | float | Yes | Warehouse longitude |
| `drivers` | array | Yes | At least 1 driver required |
| `drivers[].id` | string | Yes | External driver ID |
| `drivers[].name` | string | Yes | Driver name |
| `drivers[].vehicle_capacity_kg` | float | No | Vehicle capacity (default: 100) |
| `drivers[].preferred_language` | string | No | Language: en, ta, hi, te, kn (default: en) |
| `packages` | array | Yes | At least 1 package required |
| `packages[].id` | string | Yes | External package ID |
| `packages[].weight_kg` | float | Yes | Package weight (min: 0.01) |
| `packages[].fragility_level` | int | No | Fragility 1-5 (default: 1) |
| `packages[].address` | string | Yes | Delivery address |
| `packages[].latitude` | float | Yes | Delivery latitude |
| `packages[].longitude` | float | Yes | Delivery longitude |
| `packages[].priority` | string | No | NORMAL, HIGH, or EXPRESS (default: NORMAL) |

---

## Response Schema: AllocationResponse

```json
{
  "allocation_run_id": "550e8400-e29b-41d4-a716-446655440000",
  "allocation_date": "2026-02-10",
  "global_fairness": {
    "avg_workload": 63.2,
    "std_dev": 18.4,
    "gini_index": 0.29
  },
  "assignments": [
    {
      "driver_id": "550e8400-e29b-41d4-a716-446655440001",
      "driver_external_id": "external_driver_001",
      "driver_name": "Raju",
      "route_id": "550e8400-e29b-41d4-a716-446655440002",
      "workload_score": 65.3,
      "fairness_score": 0.82,
      "route_summary": {
        "num_packages": 22,
        "total_weight_kg": 48.5,
        "num_stops": 14,
        "route_difficulty_score": 2.1,
        "estimated_time_minutes": 145
      },
      "explanation": "You received a moderate route..."
    }
  ]
}
```

### Global Fairness Metrics

| Metric | Description |
|--------|-------------|
| `avg_workload` | Average workload score across all drivers |
| `std_dev` | Standard deviation of workload (lower = fairer) |
| `gini_index` | Gini coefficient 0-1 (lower = fairer distribution) |

---

## cURL Examples

### Minimal Example (2 drivers, 2 packages)

```bash
curl -X POST "http://localhost:8000/api/v1/allocate" \
  -H "Content-Type: application/json" \
  -d '{
    "allocation_date": "2026-02-10",
    "warehouse": { "lat": 12.9716, "lng": 77.5946 },
    "drivers": [
      {"id": "driver_001", "name": "Raju", "vehicle_capacity_kg": 150, "preferred_language": "en"},
      {"id": "driver_002", "name": "Priya", "vehicle_capacity_kg": 120, "preferred_language": "ta"}
    ],
    "packages": [
      {"id": "pkg_001", "weight_kg": 2.5, "fragility_level": 3, "address": "4A Ruby Apartment", "latitude": 12.97, "longitude": 77.60, "priority": "NORMAL"},
      {"id": "pkg_002", "weight_kg": 8.0, "fragility_level": 2, "address": "No. 12 Oak Street", "latitude": 12.98, "longitude": 77.61, "priority": "HIGH"}
    ]
  }'
```

### Full Example (5 drivers, 10 packages)

```bash
curl -X POST "http://localhost:8000/api/v1/allocate" \
  -H "Content-Type: application/json" \
  -d '{
    "allocation_date": "2026-02-10",
    "warehouse": { "lat": 12.9716, "lng": 77.5946 },
    "drivers": [
      {"id": "drv_001", "name": "Raju", "vehicle_capacity_kg": 150, "preferred_language": "en"},
      {"id": "drv_002", "name": "Priya", "vehicle_capacity_kg": 120, "preferred_language": "ta"},
      {"id": "drv_003", "name": "Kumar", "vehicle_capacity_kg": 180, "preferred_language": "en"},
      {"id": "drv_004", "name": "Lakshmi", "vehicle_capacity_kg": 100, "preferred_language": "hi"},
      {"id": "drv_005", "name": "Arjun", "vehicle_capacity_kg": 160, "preferred_language": "te"}
    ],
    "packages": [
      {"id": "pkg_001", "weight_kg": 2.5, "fragility_level": 3, "address": "Koramangala 3rd Main", "latitude": 12.9352, "longitude": 77.6245, "priority": "NORMAL"},
      {"id": "pkg_002", "weight_kg": 8.0, "fragility_level": 2, "address": "Indiranagar 2nd Cross", "latitude": 12.9716, "longitude": 77.6411, "priority": "HIGH"},
      {"id": "pkg_003", "weight_kg": 1.5, "fragility_level": 1, "address": "Brigade Road", "latitude": 12.9719, "longitude": 77.6074, "priority": "NORMAL"},
      {"id": "pkg_004", "weight_kg": 5.0, "fragility_level": 4, "address": "MG Road", "latitude": 12.9756, "longitude": 77.6066, "priority": "EXPRESS"},
      {"id": "pkg_005", "weight_kg": 3.2, "fragility_level": 2, "address": "Residency Road", "latitude": 12.9682, "longitude": 77.5973, "priority": "NORMAL"},
      {"id": "pkg_006", "weight_kg": 6.8, "fragility_level": 3, "address": "Commercial Street", "latitude": 12.9833, "longitude": 77.6072, "priority": "HIGH"},
      {"id": "pkg_007", "weight_kg": 2.1, "fragility_level": 1, "address": "Cunningham Road", "latitude": 12.9927, "longitude": 77.5855, "priority": "NORMAL"},
      {"id": "pkg_008", "weight_kg": 4.5, "fragility_level": 5, "address": "Lavelle Road", "latitude": 12.9644, "longitude": 77.5957, "priority": "EXPRESS"},
      {"id": "pkg_009", "weight_kg": 7.2, "fragility_level": 2, "address": "Vittal Mallya Road", "latitude": 12.9738, "longitude": 77.5956, "priority": "NORMAL"},
      {"id": "pkg_010", "weight_kg": 1.8, "fragility_level": 1, "address": "Infantry Road", "latitude": 12.9864, "longitude": 77.5961, "priority": "HIGH"}
    ]
  }'
```

---

## Demo Page

A visual demo page is available at:

```
http://localhost:8000/demo/allocate
```

Features:
- JSON input editor with pre-filled example
- One-click allocation execution
- Real-time metrics display (Gini, StdDev, Avg Workload)
- Response JSON viewer
- Copy-paste cURL examples

---

## Notes for External Integrators

1. **Driver IDs**: Use your own external IDs. The system tracks them via `driver_external_id` in responses.

2. **Package IDs**: Similarly, package IDs map to your system.

3. **Learning Agent**: The system internally uses historical data from previous allocations to improve fairness. This is DB-backed and doesn't require extra input fields.

4. **Idempotency**: Each allocation creates a new `allocation_run_id`. Re-running with the same data creates a new run.

5. **Error Handling**: On validation errors, you'll receive HTTP 400 with a `detail` field explaining the issue.

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/allocate` | Main allocation endpoint |
| POST | `/api/v1/allocate/langgraph` | LangGraph-based allocation (same interface) |
| GET | `/api/v1/drivers/{id}` | Get driver details |
| GET | `/api/v1/routes/{id}` | Get route details |
| POST | `/api/v1/feedback` | Submit driver feedback |
| GET | `/demo/allocate` | Visual demo page |
| GET | `/health` | Health check |

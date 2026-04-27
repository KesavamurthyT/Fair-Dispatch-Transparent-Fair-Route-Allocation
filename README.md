<p align="center">
  <b>🚚 Fair Dispatch System - Transparent & Fair Route Allocation</b><br>
  <i>A backend API for fair, explainable route allocations using a multi-agent architecture</i>
</p>

---

<div align="center">

# 📦 Fair Dispatch

*`"Fairness guaranteed." - Because unorganized dispatch leaves a trail of poor deliveries and overworked fleets`*

**Route Clustering • Workload Scoring • Fairness Metrics • Explainable Algorithms**

[![Python 3.11+](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/postgresql-4169e1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)

</div>

<br>

## Problem

Delivery operations often face challenges with unfair or unbalanced route allocations. Traditional systems focus primarily on efficiency, often leaving some drivers with overworked schedules while others are underutilized. This imbalance leads to dissatisfaction, higher turnover rates, and opaque assignment processes.

## Solution

A fairness-focused route allocation system for delivery operations. This backend API accepts packages and drivers, performs route clustering, calculates workload scores and fairness metrics, and returns fair, explainable route allocations using a **multi-agent architecture**.

### Algorithms

#### Workload Score Formula

```
workload_score = a × num_packages + b × total_weight_kg + c × route_difficulty_score + d × estimated_time_minutes
```

#### Gini Index

Measures inequality in workload distribution (0 = perfect equality, 1 = maximum inequality).

```
G = (2 × Σ(i × x_i)) / (n × Σx_i) - (n + 1) / n
```

#### Fairness Score

Per-driver fairness relative to average:

```
fairness_score = 1 - |workload - avg_workload| / max(avg_workload, 1)
```

## Features

- **Route Clustering**: Groups packages using K-Means based on geographic proximity
- **Workload Scoring**: Calculates balanced workload metrics for each route
- **Fairness Metrics**: Computes Gini index and individual fairness scores
- **Multi-Agent Optimization** (Phase 4.1):
  - **ML Effort Agent**: Builds effort matrix for driver-route pairs
  - **Route Planner Agent**: Uses OR-Tools/Hungarian for optimal assignment
  - **Fairness Manager Agent**: Evaluates fairness and triggers re-optimization
- **Explainability Engine**: Generates human-readable explanations for allocations
- **Decision Logging**: Records agent workflow for audit and visualization

## Tech Stack

- **Python 3.11+**
- **FastAPI** - Modern web framework
- **PostgreSQL** - Database (via SQLAlchemy async)
- **Alembic** - Database migrations
- **scikit-learn** - K-Means clustering
- **SciPy** - Hungarian algorithm for assignment

## Architecture diagram

The allocation pipeline uses three specialized agents:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  MLEffortAgent  │ → │ RoutePlannerAgent │ → │ FairnessManagerAgent│
│  (Effort Matrix)│    │ (Proposal 1)      │    │ (Check Fairness)    │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                          ↓
                                                   ACCEPT or REOPTIMIZE
                                                          ↓
                                               ┌──────────────────┐
                                               │ RoutePlannerAgent │
                                               │ (Proposal 2)      │
                                               └──────────────────┘
```

### Agents

| Agent | Purpose | Key Outputs |
|-------|---------|-------------|
| **MLEffortAgent** | Builds effort matrix for all driver-route pairs | Effort matrix, breakdown per pair |
| **RoutePlannerAgent** | Solves optimal assignment (OR-Tools/Hungarian) | Assignments with total effort |
| **FairnessManagerAgent** | Evaluates Gini, std dev, max gap vs thresholds | ACCEPT or REOPTIMIZE with penalties |

### Decision Logging

Each agent step is logged to the `decision_logs` table:

```sql
SELECT agent_name, step_type, input_snapshot, output_snapshot
FROM decision_logs
WHERE allocation_run_id = 'your-run-id'
ORDER BY created_at;
```

### AllocationRun Status Flow

```
PENDING → SUCCESS (or FAILED if error)
```

Query allocation runs with fairness metrics:

```sql
SELECT id, date, status, global_gini_index, global_std_dev, global_max_gap
FROM allocation_runs
ORDER BY started_at DESC LIMIT 10;
```

### Project Structure

```
fair-dispatch-system/
├── alembic/                    # Database migrations
│   ├── versions/
│   │   └── 001_initial_schema.py
│   └── env.py
├── app/
│   ├── api/                    # FastAPI routers
│   │   ├── allocation.py       # POST /api/v1/allocate
│   │   ├── drivers.py          # GET /api/v1/drivers/{id}
│   │   ├── routes.py           # GET /api/v1/routes/{id}
│   │   └── feedback.py         # POST /api/v1/feedback
│   ├── models/                 # SQLAlchemy models
│   │   ├── driver.py
│   │   ├── package.py
│   │   ├── route.py
│   │   └── assignment.py
│   ├── schemas/                # Pydantic DTOs
│   │   ├── allocation.py
│   │   ├── driver.py
│   │   ├── route.py
│   │   └── feedback.py
│   ├── services/               # Business logic
│   │   ├── clustering.py       # K-Means clustering
│   │   ├── workload.py         # Workload scoring
│   │   ├── fairness.py         # Gini index & fairness
│   │   ├── allocation.py       # Hungarian algorithm
│   │   └── explainability.py   # Explanation generator
│   ├── config.py               # Settings
│   ├── database.py             # DB connection
│   └── main.py                 # FastAPI app
├── tests/                      # Unit tests
├── .env.example
├── alembic.ini
├── requirements.txt
└── README.md
```

## Setup / Installation

### Quick Start

#### 1. Install Dependencies

```bash
cd fair-dispatch-system
pip install -r requirements.txt
```

#### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your PostgreSQL connection
# DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/fair_dispatch
```

#### 3. Create Database

```bash
# Create PostgreSQL database
createdb fair_dispatch
```

#### 4. Run Migrations

```bash
alembic upgrade head
```

#### 5. Start the Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- **Demo Page**: `http://localhost:8000/demo/allocate`

### Configuration

Configuration is managed via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `DEBUG` | `true` | Enable debug mode |
| `WORKLOAD_WEIGHT_A` | `1.0` | Weight for num_packages |
| `WORKLOAD_WEIGHT_B` | `0.5` | Weight for total_weight_kg |
| `WORKLOAD_WEIGHT_C` | `10.0` | Weight for route_difficulty_score |
| `WORKLOAD_WEIGHT_D` | `0.2` | Weight for estimated_time_minutes |
| `TARGET_PACKAGES_PER_ROUTE` | `20` | Target packages per route cluster |

### Running Tests

#### Automation

The project includes a comprehensive Makefile for running tests and setting up the environment.

```bash
# Run all tests (unit + E2E + integration)
make test

# Run E2E tests only
make test-e2e

# Run tests in parallel (faster)
make test-parallel

# Run tests with coverage report
make test-cov

# Full CI pipeline verify
make ci
```

#### Manual Setup

If you prefer running manual commands:

1. **Start Test Database** (Optional, defaults to in-memory SQLite):
   ```bash
   docker-compose -f tests/docker-compose.test.yml up -d
   export TEST_DATABASE_URL=postgresql+asyncpg://test:test@localhost:5433/fair_dispatch_test
   ```

2. **Run Pytest**:
   ```bash
   pytest tests/ -v
   ```

### Demo Page

A visual demo page for testing the allocation API with:

- **JSON Input Panel**: Pre-filled with sample `AllocationRequest` (5 drivers, 10 packages)
- **JSON Output Panel**: Displays `AllocationResponse` after allocation
- **Metrics Bar**: Shows Gini index, std deviation, avg workload, and assignment count
- **cURL Example**: Copy-paste ready command for CLI testing

See [`docs/integration.md`](docs/integration.md) for complete API integration documentation.

### API Endpoints

#### POST /api/v1/allocate

Allocate packages to drivers fairly.

**Request:**
```json
{
  "date": "2026-02-10",
  "warehouse": {"lat": 12.9716, "lng": 77.5946},
  "packages": [
    {
      "id": "pkg_001",
      "weight_kg": 2.5,
      "fragility_level": 3,
      "address": "123 Main St, Area, City",
      "latitude": 12.97,
      "longitude": 77.60,
      "priority": "NORMAL"
    },
    {
      "id": "pkg_002",
      "weight_kg": 1.0,
      "fragility_level": 1,
      "address": "456 Oak Ave, Area, City",
      "latitude": 12.98,
      "longitude": 77.61,
      "priority": "HIGH"
    }
  ],
  "drivers": [
    {
      "id": "driver_001",
      "name": "Raju",
      "vehicle_capacity_kg": 150,
      "preferred_language": "en"
    },
    {
      "id": "driver_002",
      "name": "Kumar",
      "vehicle_capacity_kg": 200,
      "preferred_language": "ta"
    }
  ]
}
```

**Response:**
```json
{
  "allocation_run_id": "550e8400-e29b-41d4-a716-446655440000",
  "date": "2026-02-10",
  "global_fairness": {
    "avg_workload": 63.2,
    "std_dev": 18.4,
    "gini_index": 0.29
  },
  "assignments": [
    {
      "driver_id": "...",
      "driver_external_id": "driver_001",
      "driver_name": "Raju",
      "route_id": "...",
      "workload_score": 65.3,
      "fairness_score": 0.82,
      "route_summary": {
        "num_packages": 22,
        "total_weight_kg": 48.5,
        "num_stops": 14,
        "route_difficulty_score": 2.1,
        "estimated_time_minutes": 145
      },
      "explanation": "Your route has 22 packages (48.5kg), 14 stops, and moderate difficulty..."
    }
  ]
}
```

#### GET /api/v1/drivers/{id}

Get driver details and recent statistics.

#### GET /api/v1/routes/{id}

Get route details and assignment information.

#### POST /api/v1/feedback

Submit driver feedback for an assignment.

```json
{
  "driver_id": "driver-uuid",
  "assignment_id": "assignment-uuid",
  "fairness_rating": 4,
  "stress_level": 5,
  "tiredness_level": 3,
  "hardest_aspect": "traffic",
  "comments": "Route was good overall"
}
```

### curl Examples

#### Allocate Packages

```bash
curl -X POST http://localhost:8000/api/v1/allocate \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-02-10",
    "warehouse": {"lat": 12.9716, "lng": 77.5946},
    "packages": [
      {"id": "pkg_001", "weight_kg": 2.5, "fragility_level": 3, "address": "123 Main St", "latitude": 12.97, "longitude": 77.60, "priority": "NORMAL"},
      {"id": "pkg_002", "weight_kg": 1.0, "fragility_level": 1, "address": "456 Oak Ave", "latitude": 12.98, "longitude": 77.61, "priority": "HIGH"}
    ],
    "drivers": [
      {"id": "driver_001", "name": "Raju", "vehicle_capacity_kg": 150, "preferred_language": "en"}
    ]
  }'
```

#### Get Driver Details

```bash
curl http://localhost:8000/api/v1/drivers/{driver_uuid}
```

#### Get Route Details

```bash
curl http://localhost:8000/api/v1/routes/{route_uuid}
```

#### Submit Feedback

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "driver_id": "driver-uuid",
    "assignment_id": "assignment-uuid",
    "fairness_rating": 4,
    "stress_level": 5,
    "tiredness_level": 3,
    "hardest_aspect": "traffic",
    "comments": "Good route today"
  }'
```

## Future Work

- **Real-Time Traffic Integration**: Incorporate live traffic data (e.g., Google Maps API) for dynamic route difficulty scoring.
- **Mobile Application**: A dedicated driver app to receive assignments, view explainability metrics, and submit instant feedback.
- **Predictive Analytics**: Utilize historical delivery data to better predict delivery times and potential delays.
- **Enhanced Multi-Agent Interactions**: Expand agents to negotiate workloads dynamically during unexpected disruptions.

## License

MIT

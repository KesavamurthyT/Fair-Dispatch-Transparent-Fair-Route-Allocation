
import pytest
import asyncio
import os
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from httpx import AsyncClient, ASGITransport

from app.database import Base, get_db
from app.main import app
from app.models import Driver, Route, DriverStatsDaily, FairnessConfig
from app.models.driver import VehicleType, PreferredLanguage
from tests.fixtures.test_data import generate_drivers, generate_routes, generate_allocation_request
from datetime import date, timedelta
import numpy as np

# Use in-memory SQLite for tests by default, unless TEST_DATABASE_URL is set
TEST_DB_URL = os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")

@pytest.fixture(scope="session")
async def test_engine():
    """Session-scoped test database engine."""
    engine = create_async_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False} if "sqlite" in TEST_DB_URL else {},
        poolclass=StaticPool if "sqlite" in TEST_DB_URL else None,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    yield engine
    
    # Drop tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Function-scoped fresh DB session."""
    connection = await test_engine.connect()
    transaction = await connection.begin()
    
    session_maker = async_sessionmaker(
        bind=connection,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    session = session_maker()
    
    yield session
    
    await session.close()
    await transaction.rollback()
    await connection.close()

@pytest.fixture
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """Test client with override for get_db."""
    async def override_get_db():
        yield db_session
        
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
        
    app.dependency_overrides.clear()

@pytest.fixture
async def sample_drivers(db_session):
    """50 drivers: 20% EV, mixed experience/stress."""
    driver_data = generate_drivers(count=50, ev_ratio=0.2)
    drivers = []
    
    for d_data in driver_data:
        driver = Driver(
            external_id=d_data["id"],
            name=d_data["name"],
            vehicle_capacity_kg=d_data["vehicle_capacity_kg"],
            preferred_language=PreferredLanguage(d_data["preferred_language"]),
            vehicle_type=VehicleType.EV if d_data["is_ev"] else VehicleType.ICE,
            battery_range_km=d_data["battery_range_km"],
            charging_time_minutes=d_data["charging_time_minutes"],
        )
        db_session.add(driver)
        drivers.append(driver)
        
    await db_session.commit()
    return drivers

@pytest.fixture
async def sample_routes(db_session):
    """50 routes: varied difficulty."""
    # We create routes in DB but allocation request will likely overwrite/use them
    # Actually allocation creates routes from packages.
    # This fixture is useful if we want to test other things that need existing routes.
    # For allocation endpoint testing, we mainly need the request object.
    
    route_data = generate_routes(count=50)
    routes = []
    
    for r_data in route_data:
        route = Route(
            date=date.today(),
            cluster_id=r_data["cluster_id"],
            total_weight_kg=r_data["total_weight_kg"],
            num_packages=r_data["stops"] * 2, 
            num_stops=r_data["stops"],
            route_difficulty_score=r_data["parking_difficulty"], 
            estimated_time_minutes=r_data["estimated_time_minutes"],
            total_distance_km=r_data["total_distance_km"]
        )
        db_session.add(route)
        routes.append(route)
        
    await db_session.commit()
    return routes

@pytest.fixture
async def allocation_request(sample_drivers):
    """Complete allocation request payload."""
    # We use sample_drivers to ensure IDs match
    
    drivers_list = []
    for d in sample_drivers:
        drivers_list.append({
            "id": d.external_id,
            "name": d.name,
            "vehicle_capacity_kg": d.vehicle_capacity_kg,
            "preferred_language": d.preferred_language.value,
            "is_ev": d.vehicle_type == VehicleType.EV
        })
    
    # Generate fresh routes for the request
    route_data = generate_routes(count=len(drivers_list))
    
    return generate_allocation_request(drivers_list, route_data)

@pytest.fixture
async def active_config(db_session):
    """Active fairness configuration."""
    config = FairnessConfig(
        is_active=True,
        gini_threshold=0.35,
        stddev_threshold=25.0,
        max_gap_threshold=25.0,
        recovery_mode_enabled=True,
        ev_safety_margin_pct=10.0,
        ev_charging_penalty_weight=0.3,
        recovery_penalty_weight=3.0
    )
    db_session.add(config)
    await db_session.commit()
    return config

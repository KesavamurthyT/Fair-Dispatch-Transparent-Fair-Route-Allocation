"""
Tests for Phase 2 Driver-facing API endpoints.
"""

import pytest
from datetime import date, timedelta
from uuid import uuid4


class TestDriverApiSchemas:
    """Tests for driver API schema validation."""
    
    def test_delivery_log_request_valid_status(self):
        """Valid delivery status should be accepted."""
        from app.schemas.driver_api import DeliveryLogRequest
        
        request = DeliveryLogRequest(
            assignment_id=uuid4(),
            route_id=uuid4(),
            driver_id=uuid4(),
            stop_order=1,
            status="DELIVERED",
            issue_type="NONE",
        )
        assert request.status == "DELIVERED"
    
    def test_stop_issue_request_validation(self):
        """Stop issue request should require notes."""
        from app.schemas.driver_api import StopIssueRequest
        
        request = StopIssueRequest(
            assignment_id=uuid4(),
            route_id=uuid4(),
            driver_id=uuid4(),
            stop_order=5,
            issue_type="SAFETY",
            notes="Dark street, no lights",
        )
        assert request.issue_type == "SAFETY"
        assert len(request.notes) > 0
    
    def test_route_swap_request_creation(self):
        """Route swap request should allow optional fields."""
        from app.schemas.driver_api import RouteSwapRequestCreate
        
        request = RouteSwapRequestCreate(
            from_driver_id=uuid4(),
            assignment_id=uuid4(),
            reason="Had heavy routes for 3 days",
            to_driver_id=None,
            preferred_date=None,
        )
        assert request.to_driver_id is None
        assert request.preferred_date is None


class TestDriverStatsResponse:
    """Tests for driver stats response schemas."""
    
    def test_day_stats_model(self):
        """DayStats should handle optional fields."""
        from app.schemas.driver_api import DayStats
        
        stats = DayStats(
            date=date.today(),
            workload_score=65.5,
            fairness_score=0.82,
            reported_stress_level=None,
            reported_fairness_rating=None,
        )
        assert stats.workload_score == 65.5
        assert stats.reported_stress_level is None
    
    def test_stats_aggregates_model(self):
        """StatsAggregates should compute correctly."""
        from app.schemas.driver_api import StatsAggregates
        
        aggregates = StatsAggregates(
            avg_workload=63.2,
            avg_fairness_score=0.83,
            avg_stress_level=3.5,
        )
        assert aggregates.avg_workload == 63.2


class TestTodayAssignmentResponse:
    """Tests for today's assignment response schema."""
    
    def test_full_response_structure(self):
        """Response should include all required nested objects."""
        from app.schemas.driver_api import (
            TodayAssignmentResponse,
            DriverDetail,
            AssignmentDetail,
            RouteSummaryDetail,
            StopDetail,
            PackageDetail,
        )
        
        driver_id = uuid4()
        route_id = uuid4()
        pkg_id = uuid4()
        
        response = TodayAssignmentResponse(
            date=date.today(),
            driver=DriverDetail(
                id=driver_id,
                external_id="ext_001",
                name="Raju",
                preferred_language="en",
            ),
            assignment=AssignmentDetail(
                assignment_id=uuid4(),
                route_id=route_id,
                workload_score=65.3,
                fairness_score=0.82,
                explanation="Moderate route assigned",
                route_summary=RouteSummaryDetail(
                    num_packages=22,
                    total_weight_kg=48.5,
                    num_stops=14,
                    route_difficulty_score=2.1,
                    estimated_time_minutes=145,
                ),
                stops=[
                    StopDetail(
                        stop_order=1,
                        address="Some Street, Area, City",
                        latitude=12.97,
                        longitude=77.60,
                        packages=[
                            PackageDetail(
                                id=pkg_id,
                                external_id="pkg_001",
                                weight_kg=2.5,
                                fragility_level=3,
                                priority="NORMAL",
                            )
                        ],
                    )
                ],
            ),
        )
        
        assert response.driver.name == "Raju"
        assert len(response.assignment.stops) == 1
        assert response.assignment.stops[0].packages[0].weight_kg == 2.5


class TestExtendedFeedbackRequest:
    """Tests for extended feedback request."""
    
    def test_extended_fields(self):
        """Extended feedback should accept new Phase 2 fields."""
        from app.schemas.driver_api import ExtendedFeedbackRequest
        
        request = ExtendedFeedbackRequest(
            driver_id=uuid4(),
            assignment_id=uuid4(),
            fairness_rating=4,
            stress_level=5,
            tiredness_level=3,
            hardest_aspect="stairs",
            route_difficulty_self_report=4,
            would_take_similar_route_again=True,
            most_unfair_aspect="parking",
            comments="Too many apartments with no lift",
        )
        
        assert request.route_difficulty_self_report == 4
        assert request.would_take_similar_route_again is True
        assert request.most_unfair_aspect == "parking"

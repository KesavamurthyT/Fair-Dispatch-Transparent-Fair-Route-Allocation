"""
Feedback API endpoint.
Handles POST /api/v1/feedback for driver feedback submission.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Driver, Assignment, DriverFeedback
from app.models.driver import HardestAspect
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit driver feedback",
    description="Allows drivers to submit feedback about their assignment.",
)
async def submit_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """Submit driver feedback for an assignment."""
    
    # Verify driver exists
    result = await db.execute(
        select(Driver).where(Driver.id == request.driver_id)
    )
    driver = result.scalar_one_or_none()
    
    if not driver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Driver with ID {request.driver_id} not found",
        )
    
    # Verify assignment exists and belongs to driver
    result = await db.execute(
        select(Assignment).where(Assignment.id == request.assignment_id)
    )
    assignment = result.scalar_one_or_none()
    
    if not assignment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment with ID {request.assignment_id} not found",
        )
    
    if assignment.driver_id != request.driver_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Assignment does not belong to the specified driver",
        )
    
    # Check for duplicate feedback
    result = await db.execute(
        select(DriverFeedback)
        .where(DriverFeedback.driver_id == request.driver_id)
        .where(DriverFeedback.assignment_id == request.assignment_id)
    )
    existing_feedback = result.scalar_one_or_none()
    
    if existing_feedback:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feedback already submitted for this assignment",
        )
    
    # Create feedback
    hardest_aspect = None
    if request.hardest_aspect:
        try:
            hardest_aspect = HardestAspect(request.hardest_aspect)
        except ValueError:
            hardest_aspect = HardestAspect.OTHER
    
    feedback = DriverFeedback(
        driver_id=request.driver_id,
        assignment_id=request.assignment_id,
        fairness_rating=request.fairness_rating,
        stress_level=request.stress_level,
        tiredness_level=request.tiredness_level,
        hardest_aspect=hardest_aspect,
        comments=request.comments,
    )
    
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    
    return FeedbackResponse(
        id=feedback.id,
        driver_id=feedback.driver_id,
        assignment_id=feedback.assignment_id,
        fairness_rating=feedback.fairness_rating,
        stress_level=feedback.stress_level,
        tiredness_level=feedback.tiredness_level,
        hardest_aspect=feedback.hardest_aspect.value if feedback.hardest_aspect else None,
        comments=feedback.comments,
        created_at=feedback.created_at,
        message="Feedback submitted successfully",
    )

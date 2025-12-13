"""
DORA Article 13 - Learning and Evolving Module.

This module implements the learning and evolving requirements as specified in
DORA Article 13, covering post-incident reviews, lessons learned documentation,
knowledge management, and continuous improvement mechanisms.

DORA Article 13 Requirements:
- Post-incident review processes
- Root cause analysis procedures
- Lessons learned documentation
- Knowledge base management
- Training needs identification
- Process improvement tracking
- Information sharing mechanisms

Reference: Regulation (EU) 2022/2554, Article 13
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from pathlib import Path
import threading
import json
import uuid


class ReviewType(Enum):
    """Types of post-incident reviews per Article 13."""
    POST_INCIDENT = "post_incident"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    TREND_ANALYSIS = "trend_analysis"
    NEAR_MISS = "near_miss"
    EXTERNAL_EVENT = "external_event"
    REGULATORY = "regulatory"
    PERIODIC = "periodic"


class LessonCategory(Enum):
    """Categories of lessons learned."""
    TECHNICAL = "technical"
    PROCEDURAL = "procedural"
    ORGANIZATIONAL = "organizational"
    COMMUNICATION = "communication"
    RESOURCE = "resource"
    VENDOR = "vendor"
    HUMAN_FACTOR = "human_factor"
    REGULATORY = "regulatory"


class LessonPriority(Enum):
    """Priority levels for lessons learned."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LessonStatus(Enum):
    """Status of lessons learned implementation."""
    IDENTIFIED = "identified"
    DOCUMENTED = "documented"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    CLOSED = "closed"


class ImprovementType(Enum):
    """Types of improvements."""
    PROCESS = "process"
    TECHNOLOGY = "technology"
    TRAINING = "training"
    DOCUMENTATION = "documentation"
    CONTROL = "control"
    POLICY = "policy"
    PROCEDURE = "procedure"
    TOOL = "tool"


class ImprovementStatus(Enum):
    """Status of improvement initiatives."""
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    REJECTED = "rejected"


class KnowledgeType(Enum):
    """Types of knowledge artifacts."""
    INCIDENT_REPORT = "incident_report"
    POST_MORTEM = "post_mortem"
    RUNBOOK = "runbook"
    PLAYBOOK = "playbook"
    BEST_PRACTICE = "best_practice"
    GUIDELINE = "guideline"
    TEMPLATE = "template"
    CASE_STUDY = "case_study"
    THREAT_INTELLIGENCE = "threat_intelligence"


@dataclass
class PostIncidentReview:
    """
    Post-incident review record per Article 13(1).

    Documents comprehensive review of ICT-related incidents including
    root cause analysis and improvement recommendations.
    """
    review_id: str
    incident_id: str
    review_type: ReviewType
    review_date: datetime
    reviewer: str
    incident_summary: str
    timeline: list[dict]
    root_causes: list[str]
    contributing_factors: list[str]
    impact_assessment: dict
    response_effectiveness: dict
    recovery_effectiveness: dict
    what_went_well: list[str]
    what_went_wrong: list[str]
    recommendations: list[str]
    action_items: list[dict]
    participants: list[str] = field(default_factory=list)
    evidence_collected: list[str] = field(default_factory=list)
    related_incidents: list[str] = field(default_factory=list)
    management_body_notified: bool = False
    notification_date: datetime | None = None
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class LessonLearned:
    """
    Lesson learned record per Article 13(2).

    Captures specific lessons from incidents for organizational learning
    and process improvement.
    """
    lesson_id: str
    title: str
    description: str
    category: LessonCategory
    priority: LessonPriority
    status: LessonStatus
    source_incident_id: str | None
    source_review_id: str | None
    identified_by: str
    identified_date: datetime
    problem_statement: str
    current_state: str
    desired_state: str
    gap_analysis: str
    recommended_actions: list[str]
    affected_areas: list[str]
    affected_systems: list[str]
    stakeholders: list[str]
    implementation_plan: dict | None = None
    implementation_date: datetime | None = None
    verified_by: str | None = None
    verification_date: datetime | None = None
    effectiveness_rating: float | None = None
    knowledge_article_id: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementInitiative:
    """
    Improvement initiative record per Article 13(3).

    Tracks continuous improvement efforts derived from lessons learned
    and post-incident reviews.
    """
    initiative_id: str
    title: str
    description: str
    improvement_type: ImprovementType
    status: ImprovementStatus
    source_lessons: list[str]
    source_reviews: list[str]
    business_justification: str
    expected_benefits: list[str]
    success_criteria: list[str]
    owner: str
    sponsor: str
    priority: LessonPriority
    estimated_effort: str
    estimated_cost: float
    actual_cost: float | None = None
    start_date: datetime | None = None
    target_completion: datetime | None = None
    actual_completion: datetime | None = None
    milestones: list[dict] = field(default_factory=list)
    resources_required: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=list)
    verification_method: str | None = None
    effectiveness_measure: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeArticle:
    """
    Knowledge base article per Article 13(4).

    Stores organizational knowledge derived from incidents, lessons,
    and best practices for future reference.
    """
    article_id: str
    title: str
    content: str
    knowledge_type: KnowledgeType
    category: str
    subcategory: str | None
    author: str
    version: str
    status: str
    applicable_systems: list[str]
    applicable_scenarios: list[str]
    related_incidents: list[str]
    related_lessons: list[str]
    prerequisites: list[str]
    procedures: list[dict]
    attachments: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    view_count: int = 0
    usefulness_rating: float = 0.0
    feedback_count: int = 0
    review_date: datetime | None = None
    expiry_date: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingNeed:
    """
    Training need identified from incidents per Article 13(5).

    Tracks training requirements identified through incident analysis
    and lessons learned.
    """
    need_id: str
    title: str
    description: str
    category: LessonCategory
    priority: LessonPriority
    source_lesson_id: str | None
    source_review_id: str | None
    identified_by: str
    identified_date: datetime
    target_audience: list[str]
    required_competencies: list[str]
    current_skill_gap: str
    recommended_training: list[str]
    delivery_method: str
    estimated_duration: str
    training_provider: str | None = None
    training_material_id: str | None = None
    scheduled_date: datetime | None = None
    completion_date: datetime | None = None
    effectiveness_assessment: str | None = None
    status: str = "identified"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TrendAnalysis:
    """
    Trend analysis record per Article 13(1)(c).

    Documents analysis of incident patterns and trends for
    proactive improvement.
    """
    analysis_id: str
    title: str
    analysis_period_start: datetime
    analysis_period_end: datetime
    analyst: str
    analysis_date: datetime
    incidents_analyzed: int
    incident_categories: dict[str, int]
    root_cause_distribution: dict[str, int]
    impact_trends: dict
    frequency_trends: dict
    seasonal_patterns: list[str]
    emerging_threats: list[str]
    systemic_issues: list[str]
    correlations: list[dict]
    predictions: list[dict]
    recommendations: list[str]
    risk_implications: list[str]
    status: str = "draft"
    management_presented: bool = False
    presentation_date: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class InformationShare:
    """
    Information sharing record per Article 13(6).

    Tracks sharing of threat intelligence and lessons learned
    with competent authorities and industry peers.
    """
    share_id: str
    title: str
    description: str
    content_type: str
    sensitivity_level: str
    source_incident_id: str | None
    source_lesson_id: str | None
    shared_by: str
    share_date: datetime
    recipients: list[str]
    sharing_channel: str
    anonymized: bool
    content_summary: str
    full_content_reference: str | None
    regulatory_requirement: bool
    acknowledgment_received: bool = False
    acknowledgment_date: datetime | None = None
    feedback_received: str | None = None
    created_at: datetime = field(default_factory=datetime.now)


class DORALearning:
    """
    DORA Article 13 Learning and Evolving Framework.

    Implements comprehensive learning mechanisms including post-incident
    reviews, lessons learned management, knowledge base maintenance,
    continuous improvement tracking, and information sharing.

    Per Article 13 Requirements:
    - Post-incident review processes
    - Root cause analysis procedures
    - Lessons learned documentation
    - Knowledge base management
    - Continuous improvement mechanisms
    - Information sharing protocols
    - Training needs identification
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize DORA Learning Framework.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._lock = threading.RLock()

        # Post-incident reviews
        self._reviews: dict[str, PostIncidentReview] = {}

        # Lessons learned
        self._lessons: dict[str, LessonLearned] = {}

        # Improvement initiatives
        self._improvements: dict[str, ImprovementInitiative] = {}

        # Knowledge articles
        self._knowledge_articles: dict[str, KnowledgeArticle] = {}

        # Training needs
        self._training_needs: dict[str, TrainingNeed] = {}

        # Trend analyses
        self._trend_analyses: dict[str, TrendAnalysis] = {}

        # Information shares
        self._information_shares: dict[str, InformationShare] = {}

        # Incident to review mapping
        self._incident_reviews: dict[str, list[str]] = {}

        # Review callbacks
        self._review_callbacks: list[Callable] = []

        # Event log
        self._event_log: list[dict] = []

    def _log_event(self, event_type: str, details: dict) -> None:
        """Log framework events."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        self._event_log.append(event)

    # =========================================================================
    # Post-Incident Review Management (Article 13(1))
    # =========================================================================

    def create_post_incident_review(
        self,
        incident_id: str,
        review_type: ReviewType,
        reviewer: str,
        incident_summary: str,
        timeline: list[dict],
        root_causes: list[str],
        contributing_factors: list[str],
        impact_assessment: dict,
        response_effectiveness: dict,
        recovery_effectiveness: dict,
        what_went_well: list[str],
        what_went_wrong: list[str],
        recommendations: list[str],
        action_items: list[dict],
        **kwargs
    ) -> PostIncidentReview:
        """
        Create post-incident review per Article 13(1).

        Args:
            incident_id: Reference to incident being reviewed
            review_type: Type of review
            reviewer: Person conducting review
            incident_summary: Brief summary of incident
            timeline: Detailed timeline of events
            root_causes: Identified root causes
            contributing_factors: Contributing factors
            impact_assessment: Assessment of incident impact
            response_effectiveness: How effective was the response
            recovery_effectiveness: How effective was the recovery
            what_went_well: Positive aspects of handling
            what_went_wrong: Areas that need improvement
            recommendations: Improvement recommendations
            action_items: Specific action items from review
            **kwargs: Additional review attributes

        Returns:
            Created PostIncidentReview instance
        """
        with self._lock:
            review_id = f"PIR-{uuid.uuid4().hex[:8].upper()}"

            review = PostIncidentReview(
                review_id=review_id,
                incident_id=incident_id,
                review_type=review_type,
                review_date=datetime.now(),
                reviewer=reviewer,
                incident_summary=incident_summary,
                timeline=timeline,
                root_causes=root_causes,
                contributing_factors=contributing_factors,
                impact_assessment=impact_assessment,
                response_effectiveness=response_effectiveness,
                recovery_effectiveness=recovery_effectiveness,
                what_went_well=what_went_well,
                what_went_wrong=what_went_wrong,
                recommendations=recommendations,
                action_items=action_items,
                **kwargs
            )

            self._reviews[review_id] = review

            # Track incident to review mapping
            if incident_id not in self._incident_reviews:
                self._incident_reviews[incident_id] = []
            self._incident_reviews[incident_id].append(review_id)

            self._log_event("review_created", {
                "review_id": review_id,
                "incident_id": incident_id,
                "review_type": review_type.value
            })

            # Trigger callbacks
            for callback in self._review_callbacks:
                try:
                    callback(review)
                except Exception:
                    pass

            return review

    def get_review(self, review_id: str) -> PostIncidentReview | None:
        """Get post-incident review by ID."""
        with self._lock:
            return self._reviews.get(review_id)

    def get_reviews_for_incident(self, incident_id: str) -> list[PostIncidentReview]:
        """Get all reviews for an incident."""
        with self._lock:
            review_ids = self._incident_reviews.get(incident_id, [])
            return [self._reviews[rid] for rid in review_ids if rid in self._reviews]

    def update_review_status(self, review_id: str, status: str, notified: bool = False) -> bool:
        """Update review status."""
        with self._lock:
            if review_id not in self._reviews:
                return False

            review = self._reviews[review_id]
            review.status = status
            review.updated_at = datetime.now()

            if notified:
                review.management_body_notified = True
                review.notification_date = datetime.now()

            self._log_event("review_status_updated", {
                "review_id": review_id,
                "status": status,
                "management_notified": notified
            })

            return True

    def add_review_callback(self, callback: Callable) -> None:
        """Register callback for new reviews."""
        self._review_callbacks.append(callback)

    # =========================================================================
    # Lessons Learned Management (Article 13(2))
    # =========================================================================

    def document_lesson_learned(
        self,
        title: str,
        description: str,
        category: LessonCategory,
        priority: LessonPriority,
        identified_by: str,
        problem_statement: str,
        current_state: str,
        desired_state: str,
        gap_analysis: str,
        recommended_actions: list[str],
        affected_areas: list[str],
        affected_systems: list[str],
        stakeholders: list[str],
        source_incident_id: str | None = None,
        source_review_id: str | None = None,
        **kwargs
    ) -> LessonLearned:
        """
        Document lesson learned per Article 13(2).

        Args:
            title: Title of lesson
            description: Detailed description
            category: Lesson category
            priority: Priority level
            identified_by: Person who identified the lesson
            problem_statement: Clear statement of the problem
            current_state: Description of current state
            desired_state: Description of desired state
            gap_analysis: Analysis of gap between states
            recommended_actions: Recommended improvement actions
            affected_areas: Business areas affected
            affected_systems: ICT systems affected
            stakeholders: Stakeholders involved
            source_incident_id: Related incident ID
            source_review_id: Related review ID
            **kwargs: Additional attributes

        Returns:
            Created LessonLearned instance
        """
        with self._lock:
            lesson_id = f"LL-{uuid.uuid4().hex[:8].upper()}"

            lesson = LessonLearned(
                lesson_id=lesson_id,
                title=title,
                description=description,
                category=category,
                priority=priority,
                status=LessonStatus.IDENTIFIED,
                source_incident_id=source_incident_id,
                source_review_id=source_review_id,
                identified_by=identified_by,
                identified_date=datetime.now(),
                problem_statement=problem_statement,
                current_state=current_state,
                desired_state=desired_state,
                gap_analysis=gap_analysis,
                recommended_actions=recommended_actions,
                affected_areas=affected_areas,
                affected_systems=affected_systems,
                stakeholders=stakeholders,
                **kwargs
            )

            self._lessons[lesson_id] = lesson

            self._log_event("lesson_documented", {
                "lesson_id": lesson_id,
                "title": title,
                "category": category.value,
                "priority": priority.value
            })

            return lesson

    def get_lesson(self, lesson_id: str) -> LessonLearned | None:
        """Get lesson learned by ID."""
        with self._lock:
            return self._lessons.get(lesson_id)

    def update_lesson_status(
        self,
        lesson_id: str,
        status: LessonStatus,
        implementation_plan: dict | None = None
    ) -> bool:
        """Update lesson learned status."""
        with self._lock:
            if lesson_id not in self._lessons:
                return False

            lesson = self._lessons[lesson_id]
            lesson.status = status
            lesson.updated_at = datetime.now()

            if implementation_plan:
                lesson.implementation_plan = implementation_plan

            if status == LessonStatus.IMPLEMENTED:
                lesson.implementation_date = datetime.now()

            self._log_event("lesson_status_updated", {
                "lesson_id": lesson_id,
                "status": status.value
            })

            return True

    def verify_lesson_implementation(
        self,
        lesson_id: str,
        verified_by: str,
        effectiveness_rating: float
    ) -> bool:
        """Verify lesson learned implementation."""
        with self._lock:
            if lesson_id not in self._lessons:
                return False

            lesson = self._lessons[lesson_id]
            lesson.status = LessonStatus.VERIFIED
            lesson.verified_by = verified_by
            lesson.verification_date = datetime.now()
            lesson.effectiveness_rating = effectiveness_rating
            lesson.updated_at = datetime.now()

            self._log_event("lesson_verified", {
                "lesson_id": lesson_id,
                "verified_by": verified_by,
                "effectiveness_rating": effectiveness_rating
            })

            return True

    def get_lessons_by_category(self, category: LessonCategory) -> list[LessonLearned]:
        """Get lessons by category."""
        with self._lock:
            return [l for l in self._lessons.values() if l.category == category]

    def get_lessons_by_status(self, status: LessonStatus) -> list[LessonLearned]:
        """Get lessons by status."""
        with self._lock:
            return [l for l in self._lessons.values() if l.status == status]

    def get_pending_lessons(self) -> list[LessonLearned]:
        """Get lessons pending implementation."""
        pending_statuses = {
            LessonStatus.IDENTIFIED,
            LessonStatus.DOCUMENTED,
            LessonStatus.REVIEWED,
            LessonStatus.APPROVED,
            LessonStatus.IN_PROGRESS
        }
        with self._lock:
            return [l for l in self._lessons.values() if l.status in pending_statuses]

    # =========================================================================
    # Improvement Initiative Management (Article 13(3))
    # =========================================================================

    def create_improvement_initiative(
        self,
        title: str,
        description: str,
        improvement_type: ImprovementType,
        source_lessons: list[str],
        source_reviews: list[str],
        business_justification: str,
        expected_benefits: list[str],
        success_criteria: list[str],
        owner: str,
        sponsor: str,
        priority: LessonPriority,
        estimated_effort: str,
        estimated_cost: float,
        **kwargs
    ) -> ImprovementInitiative:
        """
        Create improvement initiative per Article 13(3).

        Args:
            title: Initiative title
            description: Detailed description
            improvement_type: Type of improvement
            source_lessons: Related lessons learned
            source_reviews: Related reviews
            business_justification: Business case
            expected_benefits: Expected benefits
            success_criteria: Success criteria
            owner: Initiative owner
            sponsor: Executive sponsor
            priority: Priority level
            estimated_effort: Effort estimate
            estimated_cost: Cost estimate
            **kwargs: Additional attributes

        Returns:
            Created ImprovementInitiative instance
        """
        with self._lock:
            initiative_id = f"IMP-{uuid.uuid4().hex[:8].upper()}"

            initiative = ImprovementInitiative(
                initiative_id=initiative_id,
                title=title,
                description=description,
                improvement_type=improvement_type,
                status=ImprovementStatus.PROPOSED,
                source_lessons=source_lessons,
                source_reviews=source_reviews,
                business_justification=business_justification,
                expected_benefits=expected_benefits,
                success_criteria=success_criteria,
                owner=owner,
                sponsor=sponsor,
                priority=priority,
                estimated_effort=estimated_effort,
                estimated_cost=estimated_cost,
                **kwargs
            )

            self._improvements[initiative_id] = initiative

            self._log_event("improvement_created", {
                "initiative_id": initiative_id,
                "title": title,
                "improvement_type": improvement_type.value
            })

            return initiative

    def get_improvement(self, initiative_id: str) -> ImprovementInitiative | None:
        """Get improvement initiative by ID."""
        with self._lock:
            return self._improvements.get(initiative_id)

    def update_improvement_status(
        self,
        initiative_id: str,
        status: ImprovementStatus,
        actual_cost: float | None = None
    ) -> bool:
        """Update improvement initiative status."""
        with self._lock:
            if initiative_id not in self._improvements:
                return False

            initiative = self._improvements[initiative_id]
            initiative.status = status
            initiative.updated_at = datetime.now()

            if status == ImprovementStatus.IN_PROGRESS and not initiative.start_date:
                initiative.start_date = datetime.now()

            if status == ImprovementStatus.COMPLETED:
                initiative.actual_completion = datetime.now()
                if actual_cost is not None:
                    initiative.actual_cost = actual_cost

            self._log_event("improvement_status_updated", {
                "initiative_id": initiative_id,
                "status": status.value
            })

            return True

    def add_improvement_milestone(
        self,
        initiative_id: str,
        milestone: dict
    ) -> bool:
        """Add milestone to improvement initiative."""
        with self._lock:
            if initiative_id not in self._improvements:
                return False

            initiative = self._improvements[initiative_id]
            milestone["id"] = f"MS-{len(initiative.milestones) + 1}"
            milestone["created_at"] = datetime.now().isoformat()
            initiative.milestones.append(milestone)
            initiative.updated_at = datetime.now()

            return True

    def get_active_improvements(self) -> list[ImprovementInitiative]:
        """Get active improvement initiatives."""
        active_statuses = {
            ImprovementStatus.APPROVED,
            ImprovementStatus.PLANNED,
            ImprovementStatus.IN_PROGRESS
        }
        with self._lock:
            return [i for i in self._improvements.values() if i.status in active_statuses]

    # =========================================================================
    # Knowledge Base Management (Article 13(4))
    # =========================================================================

    def create_knowledge_article(
        self,
        title: str,
        content: str,
        knowledge_type: KnowledgeType,
        category: str,
        author: str,
        applicable_systems: list[str],
        applicable_scenarios: list[str],
        procedures: list[dict],
        subcategory: str | None = None,
        related_incidents: list[str] | None = None,
        related_lessons: list[str] | None = None,
        prerequisites: list[str] | None = None,
        **kwargs
    ) -> KnowledgeArticle:
        """
        Create knowledge article per Article 13(4).

        Args:
            title: Article title
            content: Article content
            knowledge_type: Type of knowledge
            category: Category classification
            author: Article author
            applicable_systems: Systems this applies to
            applicable_scenarios: Scenarios this applies to
            procedures: Step-by-step procedures
            subcategory: Optional subcategory
            related_incidents: Related incident IDs
            related_lessons: Related lesson IDs
            prerequisites: Prerequisites for using this knowledge
            **kwargs: Additional attributes

        Returns:
            Created KnowledgeArticle instance
        """
        with self._lock:
            article_id = f"KB-{uuid.uuid4().hex[:8].upper()}"

            article = KnowledgeArticle(
                article_id=article_id,
                title=title,
                content=content,
                knowledge_type=knowledge_type,
                category=category,
                subcategory=subcategory,
                author=author,
                version="1.0",
                status="draft",
                applicable_systems=applicable_systems,
                applicable_scenarios=applicable_scenarios,
                related_incidents=related_incidents or [],
                related_lessons=related_lessons or [],
                prerequisites=prerequisites or [],
                procedures=procedures,
                **kwargs
            )

            self._knowledge_articles[article_id] = article

            self._log_event("knowledge_article_created", {
                "article_id": article_id,
                "title": title,
                "knowledge_type": knowledge_type.value
            })

            return article

    def get_knowledge_article(self, article_id: str) -> KnowledgeArticle | None:
        """Get knowledge article by ID."""
        with self._lock:
            return self._knowledge_articles.get(article_id)

    def search_knowledge_base(
        self,
        query: str,
        knowledge_type: KnowledgeType | None = None,
        category: str | None = None
    ) -> list[KnowledgeArticle]:
        """
        Search knowledge base.

        Args:
            query: Search query
            knowledge_type: Optional type filter
            category: Optional category filter

        Returns:
            List of matching articles
        """
        with self._lock:
            results = []
            query_lower = query.lower()

            for article in self._knowledge_articles.values():
                # Filter by type
                if knowledge_type and article.knowledge_type != knowledge_type:
                    continue

                # Filter by category
                if category and article.category != category:
                    continue

                # Search in title, content, keywords, tags
                if (query_lower in article.title.lower() or
                    query_lower in article.content.lower() or
                    any(query_lower in kw.lower() for kw in article.keywords) or
                    any(query_lower in tag.lower() for tag in article.tags)):
                    results.append(article)

            return results

    def update_article_version(
        self,
        article_id: str,
        new_content: str,
        author: str
    ) -> bool:
        """Update knowledge article content and version."""
        with self._lock:
            if article_id not in self._knowledge_articles:
                return False

            article = self._knowledge_articles[article_id]
            article.content = new_content
            article.author = author

            # Increment version
            major, minor = article.version.split(".")
            article.version = f"{major}.{int(minor) + 1}"
            article.updated_at = datetime.now()

            self._log_event("article_updated", {
                "article_id": article_id,
                "new_version": article.version
            })

            return True

    def record_article_view(self, article_id: str) -> bool:
        """Record article view for analytics."""
        with self._lock:
            if article_id not in self._knowledge_articles:
                return False

            self._knowledge_articles[article_id].view_count += 1
            return True

    def rate_article_usefulness(
        self,
        article_id: str,
        rating: float
    ) -> bool:
        """Rate article usefulness (0-5 scale)."""
        with self._lock:
            if article_id not in self._knowledge_articles:
                return False

            article = self._knowledge_articles[article_id]
            # Calculate running average
            total = article.usefulness_rating * article.feedback_count
            article.feedback_count += 1
            article.usefulness_rating = (total + rating) / article.feedback_count

            return True

    # =========================================================================
    # Training Needs Management (Article 13(5))
    # =========================================================================

    def identify_training_need(
        self,
        title: str,
        description: str,
        category: LessonCategory,
        priority: LessonPriority,
        identified_by: str,
        target_audience: list[str],
        required_competencies: list[str],
        current_skill_gap: str,
        recommended_training: list[str],
        delivery_method: str,
        estimated_duration: str,
        source_lesson_id: str | None = None,
        source_review_id: str | None = None,
        **kwargs
    ) -> TrainingNeed:
        """
        Identify training need per Article 13(5).

        Args:
            title: Training need title
            description: Detailed description
            category: Need category
            priority: Priority level
            identified_by: Person who identified need
            target_audience: Who needs training
            required_competencies: Competencies to develop
            current_skill_gap: Description of current gap
            recommended_training: Recommended training options
            delivery_method: How to deliver training
            estimated_duration: Estimated training duration
            source_lesson_id: Related lesson
            source_review_id: Related review
            **kwargs: Additional attributes

        Returns:
            Created TrainingNeed instance
        """
        with self._lock:
            need_id = f"TN-{uuid.uuid4().hex[:8].upper()}"

            need = TrainingNeed(
                need_id=need_id,
                title=title,
                description=description,
                category=category,
                priority=priority,
                source_lesson_id=source_lesson_id,
                source_review_id=source_review_id,
                identified_by=identified_by,
                identified_date=datetime.now(),
                target_audience=target_audience,
                required_competencies=required_competencies,
                current_skill_gap=current_skill_gap,
                recommended_training=recommended_training,
                delivery_method=delivery_method,
                estimated_duration=estimated_duration,
                **kwargs
            )

            self._training_needs[need_id] = need

            self._log_event("training_need_identified", {
                "need_id": need_id,
                "title": title,
                "priority": priority.value
            })

            return need

    def get_training_need(self, need_id: str) -> TrainingNeed | None:
        """Get training need by ID."""
        with self._lock:
            return self._training_needs.get(need_id)

    def update_training_status(
        self,
        need_id: str,
        status: str,
        scheduled_date: datetime | None = None,
        completion_date: datetime | None = None,
        effectiveness_assessment: str | None = None
    ) -> bool:
        """Update training need status."""
        with self._lock:
            if need_id not in self._training_needs:
                return False

            need = self._training_needs[need_id]
            need.status = status
            need.updated_at = datetime.now()

            if scheduled_date:
                need.scheduled_date = scheduled_date
            if completion_date:
                need.completion_date = completion_date
            if effectiveness_assessment:
                need.effectiveness_assessment = effectiveness_assessment

            self._log_event("training_status_updated", {
                "need_id": need_id,
                "status": status
            })

            return True

    def get_pending_training_needs(self) -> list[TrainingNeed]:
        """Get training needs not yet completed."""
        with self._lock:
            return [n for n in self._training_needs.values()
                    if n.status not in {"completed", "cancelled"}]

    # =========================================================================
    # Trend Analysis (Article 13(1)(c))
    # =========================================================================

    def create_trend_analysis(
        self,
        title: str,
        analysis_period_start: datetime,
        analysis_period_end: datetime,
        analyst: str,
        incidents_analyzed: int,
        incident_categories: dict[str, int],
        root_cause_distribution: dict[str, int],
        impact_trends: dict,
        frequency_trends: dict,
        seasonal_patterns: list[str],
        emerging_threats: list[str],
        systemic_issues: list[str],
        correlations: list[dict],
        predictions: list[dict],
        recommendations: list[str],
        risk_implications: list[str],
        **kwargs
    ) -> TrendAnalysis:
        """
        Create trend analysis per Article 13(1)(c).

        Args:
            title: Analysis title
            analysis_period_start: Start of analysis period
            analysis_period_end: End of analysis period
            analyst: Person conducting analysis
            incidents_analyzed: Number of incidents analyzed
            incident_categories: Distribution by category
            root_cause_distribution: Distribution of root causes
            impact_trends: Impact trend data
            frequency_trends: Frequency trend data
            seasonal_patterns: Identified seasonal patterns
            emerging_threats: Emerging threat indicators
            systemic_issues: Systemic issues identified
            correlations: Correlations found
            predictions: Predictive insights
            recommendations: Recommendations
            risk_implications: Risk implications
            **kwargs: Additional attributes

        Returns:
            Created TrendAnalysis instance
        """
        with self._lock:
            analysis_id = f"TA-{uuid.uuid4().hex[:8].upper()}"

            analysis = TrendAnalysis(
                analysis_id=analysis_id,
                title=title,
                analysis_period_start=analysis_period_start,
                analysis_period_end=analysis_period_end,
                analyst=analyst,
                analysis_date=datetime.now(),
                incidents_analyzed=incidents_analyzed,
                incident_categories=incident_categories,
                root_cause_distribution=root_cause_distribution,
                impact_trends=impact_trends,
                frequency_trends=frequency_trends,
                seasonal_patterns=seasonal_patterns,
                emerging_threats=emerging_threats,
                systemic_issues=systemic_issues,
                correlations=correlations,
                predictions=predictions,
                recommendations=recommendations,
                risk_implications=risk_implications,
                **kwargs
            )

            self._trend_analyses[analysis_id] = analysis

            self._log_event("trend_analysis_created", {
                "analysis_id": analysis_id,
                "title": title,
                "incidents_analyzed": incidents_analyzed
            })

            return analysis

    def get_trend_analysis(self, analysis_id: str) -> TrendAnalysis | None:
        """Get trend analysis by ID."""
        with self._lock:
            return self._trend_analyses.get(analysis_id)

    def present_to_management(self, analysis_id: str) -> bool:
        """Mark trend analysis as presented to management."""
        with self._lock:
            if analysis_id not in self._trend_analyses:
                return False

            analysis = self._trend_analyses[analysis_id]
            analysis.management_presented = True
            analysis.presentation_date = datetime.now()
            analysis.status = "presented"
            analysis.updated_at = datetime.now()

            return True

    # =========================================================================
    # Information Sharing (Article 13(6))
    # =========================================================================

    def share_information(
        self,
        title: str,
        description: str,
        content_type: str,
        sensitivity_level: str,
        shared_by: str,
        recipients: list[str],
        sharing_channel: str,
        content_summary: str,
        anonymized: bool = True,
        source_incident_id: str | None = None,
        source_lesson_id: str | None = None,
        full_content_reference: str | None = None,
        regulatory_requirement: bool = False,
        **kwargs
    ) -> InformationShare:
        """
        Share information per Article 13(6).

        Args:
            title: Share title
            description: Description
            content_type: Type of content being shared
            sensitivity_level: Sensitivity classification
            shared_by: Person sharing
            recipients: Recipients of share
            sharing_channel: Channel used for sharing
            content_summary: Summary of content
            anonymized: Whether content is anonymized
            source_incident_id: Source incident
            source_lesson_id: Source lesson
            full_content_reference: Reference to full content
            regulatory_requirement: Whether required by regulation
            **kwargs: Additional attributes

        Returns:
            Created InformationShare instance
        """
        with self._lock:
            share_id = f"IS-{uuid.uuid4().hex[:8].upper()}"

            share = InformationShare(
                share_id=share_id,
                title=title,
                description=description,
                content_type=content_type,
                sensitivity_level=sensitivity_level,
                source_incident_id=source_incident_id,
                source_lesson_id=source_lesson_id,
                shared_by=shared_by,
                share_date=datetime.now(),
                recipients=recipients,
                sharing_channel=sharing_channel,
                anonymized=anonymized,
                content_summary=content_summary,
                full_content_reference=full_content_reference,
                regulatory_requirement=regulatory_requirement,
                **kwargs
            )

            self._information_shares[share_id] = share

            self._log_event("information_shared", {
                "share_id": share_id,
                "title": title,
                "recipients": recipients,
                "regulatory_requirement": regulatory_requirement
            })

            return share

    def get_information_share(self, share_id: str) -> InformationShare | None:
        """Get information share by ID."""
        with self._lock:
            return self._information_shares.get(share_id)

    def acknowledge_share(self, share_id: str, feedback: str | None = None) -> bool:
        """Record acknowledgment of information share."""
        with self._lock:
            if share_id not in self._information_shares:
                return False

            share = self._information_shares[share_id]
            share.acknowledgment_received = True
            share.acknowledgment_date = datetime.now()

            if feedback:
                share.feedback_received = feedback

            return True

    # =========================================================================
    # Analytics and Reporting
    # =========================================================================

    def get_learning_metrics(self) -> dict:
        """Get learning framework metrics."""
        with self._lock:
            return {
                "total_reviews": len(self._reviews),
                "reviews_by_type": self._count_by_enum(
                    self._reviews.values(), "review_type"
                ),
                "total_lessons": len(self._lessons),
                "lessons_by_category": self._count_by_enum(
                    self._lessons.values(), "category"
                ),
                "lessons_by_status": self._count_by_enum(
                    self._lessons.values(), "status"
                ),
                "lessons_by_priority": self._count_by_enum(
                    self._lessons.values(), "priority"
                ),
                "total_improvements": len(self._improvements),
                "improvements_by_type": self._count_by_enum(
                    self._improvements.values(), "improvement_type"
                ),
                "improvements_by_status": self._count_by_enum(
                    self._improvements.values(), "status"
                ),
                "total_knowledge_articles": len(self._knowledge_articles),
                "articles_by_type": self._count_by_enum(
                    self._knowledge_articles.values(), "knowledge_type"
                ),
                "total_training_needs": len(self._training_needs),
                "pending_training_needs": len(self.get_pending_training_needs()),
                "total_trend_analyses": len(self._trend_analyses),
                "total_information_shares": len(self._information_shares)
            }

    def _count_by_enum(self, items: Any, attr: str) -> dict[str, int]:
        """Count items by enum attribute."""
        counts: dict[str, int] = {}
        for item in items:
            value = getattr(item, attr)
            key = value.value if hasattr(value, "value") else str(value)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def generate_learning_report(self, period_start: datetime, period_end: datetime) -> dict:
        """
        Generate learning and evolving report.

        Args:
            period_start: Report period start
            period_end: Report period end

        Returns:
            Comprehensive learning report
        """
        with self._lock:
            # Filter items by period
            reviews_in_period = [
                r for r in self._reviews.values()
                if period_start <= r.review_date <= period_end
            ]

            lessons_in_period = [
                l for l in self._lessons.values()
                if period_start <= l.identified_date <= period_end
            ]

            improvements_in_period = [
                i for i in self._improvements.values()
                if period_start <= i.created_at <= period_end
            ]

            shares_in_period = [
                s for s in self._information_shares.values()
                if period_start <= s.share_date <= period_end
            ]

            return {
                "report_period": {
                    "start": period_start.isoformat(),
                    "end": period_end.isoformat()
                },
                "post_incident_reviews": {
                    "count": len(reviews_in_period),
                    "by_type": self._count_by_enum(reviews_in_period, "review_type"),
                    "root_causes_identified": sum(
                        len(r.root_causes) for r in reviews_in_period
                    ),
                    "recommendations_made": sum(
                        len(r.recommendations) for r in reviews_in_period
                    )
                },
                "lessons_learned": {
                    "count": len(lessons_in_period),
                    "by_category": self._count_by_enum(lessons_in_period, "category"),
                    "by_priority": self._count_by_enum(lessons_in_period, "priority"),
                    "implementation_rate": self._calculate_implementation_rate(
                        lessons_in_period
                    )
                },
                "improvement_initiatives": {
                    "count": len(improvements_in_period),
                    "by_type": self._count_by_enum(improvements_in_period, "improvement_type"),
                    "by_status": self._count_by_enum(improvements_in_period, "status"),
                    "total_investment": sum(
                        i.estimated_cost for i in improvements_in_period
                    )
                },
                "knowledge_base": {
                    "total_articles": len(self._knowledge_articles),
                    "most_viewed": self._get_top_articles(5),
                    "highest_rated": self._get_top_rated_articles(5)
                },
                "information_sharing": {
                    "count": len(shares_in_period),
                    "regulatory_shares": len(
                        [s for s in shares_in_period if s.regulatory_requirement]
                    ),
                    "acknowledgment_rate": self._calculate_acknowledgment_rate(
                        shares_in_period
                    )
                },
                "generated_at": datetime.now().isoformat()
            }

    def _calculate_implementation_rate(self, lessons: list) -> float:
        """Calculate lesson implementation rate."""
        if not lessons:
            return 0.0
        implemented = len([
            l for l in lessons
            if l.status in {LessonStatus.IMPLEMENTED, LessonStatus.VERIFIED, LessonStatus.CLOSED}
        ])
        return implemented / len(lessons)

    def _get_top_articles(self, n: int) -> list[dict]:
        """Get top N most viewed articles."""
        sorted_articles = sorted(
            self._knowledge_articles.values(),
            key=lambda a: a.view_count,
            reverse=True
        )[:n]
        return [
            {"article_id": a.article_id, "title": a.title, "views": a.view_count}
            for a in sorted_articles
        ]

    def _get_top_rated_articles(self, n: int) -> list[dict]:
        """Get top N highest rated articles."""
        rated_articles = [
            a for a in self._knowledge_articles.values()
            if a.feedback_count > 0
        ]
        sorted_articles = sorted(
            rated_articles,
            key=lambda a: a.usefulness_rating,
            reverse=True
        )[:n]
        return [
            {"article_id": a.article_id, "title": a.title, "rating": a.usefulness_rating}
            for a in sorted_articles
        ]

    def _calculate_acknowledgment_rate(self, shares: list) -> float:
        """Calculate information share acknowledgment rate."""
        if not shares:
            return 0.0
        acknowledged = len([s for s in shares if s.acknowledgment_received])
        return acknowledged / len(shares)

    # =========================================================================
    # Export and Persistence
    # =========================================================================

    def export_to_file(self, filepath: Path) -> bool:
        """Export learning data to JSONL file."""
        try:
            with open(filepath, "w") as f:
                for event in self._event_log:
                    f.write(json.dumps(event) + "\n")
            return True
        except Exception:
            return False

    def get_event_log(self) -> list[dict]:
        """Get event log."""
        with self._lock:
            return list(self._event_log)


# =============================================================================
# Factory Function
# =============================================================================

def create_dora_learning(config: dict | None = None) -> DORALearning:
    """
    Factory function to create DORA Learning Framework.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured DORALearning instance
    """
    return DORALearning(config=config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "ReviewType",
    "LessonCategory",
    "LessonPriority",
    "LessonStatus",
    "ImprovementType",
    "ImprovementStatus",
    "KnowledgeType",
    # Dataclasses
    "PostIncidentReview",
    "LessonLearned",
    "ImprovementInitiative",
    "KnowledgeArticle",
    "TrainingNeed",
    "TrendAnalysis",
    "InformationShare",
    # Main class
    "DORALearning",
    # Factory
    "create_dora_learning",
]

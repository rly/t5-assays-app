"""SQLAlchemy ORM models for users, preferences, dataset selections, and chat."""
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    preferences = relationship("UserPreference", back_populates="user", uselist=False)
    dataset_selections = relationship("DatasetSelection", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    viewing_dataset_key = Column(String, nullable=True)  # currently viewed dataset
    selected_model = Column(String, default="nvidia/nemotron-3-super-120b-a12b:free")
    openrouter_api_key_encrypted = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="preferences")


class DatasetSelection(Base):
    """Tracks each user's relationship to a dataset: filters and AI access."""
    __tablename__ = "dataset_selections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_key = Column(String, nullable=False)  # merge config key or sheet name
    dataset_type = Column(String, nullable=False)  # "merge", "sheet", "merge-source"
    display_name = Column(String, nullable=False)
    provided_to_ai = Column(Boolean, default=False)
    filters_json = Column(Text, default="{}")  # JSON: {"Chi2_ndof_RU2": 10.0}
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="dataset_selections")


class Conversation(Base):
    """One conversation per user (not per dataset)."""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    model_used = Column(String, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")

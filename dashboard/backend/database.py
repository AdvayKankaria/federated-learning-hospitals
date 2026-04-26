"""
Database Models for Metrics Storage
=====================================
SQLite-based storage for experiment history.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

Base = declarative_base()


class Experiment(Base):
    """Experiment record."""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String, default="pending")  # pending, running, completed, failed
    
    # Configuration
    num_hospitals = Column(Integer)
    num_rounds = Column(Integer)
    local_epochs = Column(Integer)
    batch_size = Column(Integer)
    learning_rate = Column(Float)
    epsilon = Column(Float)
    delta = Column(Float)
    aggregation_strategy = Column(String)
    partition_type = Column(String)
    
    # Results
    final_accuracy = Column(Float, nullable=True)
    final_auc_roc = Column(Float, nullable=True)
    best_accuracy = Column(Float, nullable=True)
    best_round = Column(Integer, nullable=True)
    total_epsilon_spent = Column(Float, nullable=True)
    
    # Full config and metrics as JSON
    config = Column(JSON, nullable=True)
    summary = Column(JSON, nullable=True)


class RoundRecord(Base):
    """Round metrics record."""
    __tablename__ = "rounds"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, index=True)
    round_num = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Metrics
    global_loss = Column(Float)
    global_accuracy = Column(Float)
    auc_roc = Column(Float, nullable=True)
    sensitivity = Column(Float, nullable=True)
    specificity = Column(Float, nullable=True)
    
    # Privacy
    epsilon_spent = Column(Float, nullable=True)
    
    # Client info
    num_clients = Column(Integer)
    num_failures = Column(Integer, default=0)
    
    # Detailed metrics as JSON
    hospital_metrics = Column(JSON, nullable=True)


class HospitalRecord(Base):
    """Hospital metrics record."""
    __tablename__ = "hospitals"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, index=True)
    hospital_id = Column(Integer, index=True)
    round_num = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Metrics
    train_loss = Column(Float)
    train_accuracy = Column(Float)
    val_loss = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    num_samples = Column(Integer)
    
    # Additional info
    metrics = Column(JSON, nullable=True)


class Database:
    """Database manager."""
    
    def __init__(self, db_path: str = "./data/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(bind=self.engine)
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> int:
        """Create new experiment record."""
        with self.get_session() as session:
            experiment = Experiment(
                name=name,
                num_hospitals=config.get("num_hospitals", 5),
                num_rounds=config.get("num_rounds", 50),
                local_epochs=config.get("local_epochs", 3),
                batch_size=config.get("batch_size", 32),
                learning_rate=config.get("learning_rate", 0.001),
                epsilon=config.get("epsilon", 8.0),
                delta=config.get("delta", 1e-5),
                aggregation_strategy=config.get("aggregation_strategy", "adaptive"),
                partition_type=config.get("partition_type", "non_iid"),
                config=config,
                status="running"
            )
            session.add(experiment)
            session.commit()
            return experiment.id
    
    def update_experiment(
        self,
        experiment_id: int,
        **kwargs
    ) -> None:
        """Update experiment record."""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter(
                Experiment.id == experiment_id
            ).first()
            
            if experiment:
                for key, value in kwargs.items():
                    if hasattr(experiment, key):
                        setattr(experiment, key, value)
                session.commit()
    
    def complete_experiment(
        self,
        experiment_id: int,
        final_accuracy: float,
        final_auc_roc: float,
        best_accuracy: float,
        best_round: int,
        total_epsilon_spent: float,
        summary: Dict = None
    ) -> None:
        """Mark experiment as completed."""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter(
                Experiment.id == experiment_id
            ).first()
            
            if experiment:
                experiment.status = "completed"
                experiment.completed_at = datetime.utcnow()
                experiment.final_accuracy = final_accuracy
                experiment.final_auc_roc = final_auc_roc
                experiment.best_accuracy = best_accuracy
                experiment.best_round = best_round
                experiment.total_epsilon_spent = total_epsilon_spent
                experiment.summary = summary
                session.commit()
    
    def add_round_metrics(
        self,
        experiment_id: int,
        round_num: int,
        metrics: Dict[str, Any],
        hospital_metrics: Dict[str, Dict] = None
    ) -> None:
        """Add round metrics."""
        with self.get_session() as session:
            record = RoundRecord(
                experiment_id=experiment_id,
                round_num=round_num,
                global_loss=metrics.get("global_loss", metrics.get("test_loss", 0)),
                global_accuracy=metrics.get("global_accuracy", metrics.get("test_accuracy", 0)),
                auc_roc=metrics.get("auc_roc", metrics.get("test_auc_roc")),
                sensitivity=metrics.get("sensitivity", metrics.get("test_sensitivity")),
                specificity=metrics.get("specificity", metrics.get("test_specificity")),
                epsilon_spent=metrics.get("epsilon_spent"),
                num_clients=metrics.get("num_clients", 0),
                num_failures=metrics.get("num_failures", 0),
                hospital_metrics=hospital_metrics
            )
            session.add(record)
            session.commit()
    
    def add_hospital_metrics(
        self,
        experiment_id: int,
        hospital_id: int,
        round_num: int,
        metrics: Dict[str, Any]
    ) -> None:
        """Add hospital-specific metrics."""
        with self.get_session() as session:
            record = HospitalRecord(
                experiment_id=experiment_id,
                hospital_id=hospital_id,
                round_num=round_num,
                train_loss=metrics.get("train_loss", 0),
                train_accuracy=metrics.get("train_accuracy", 0),
                val_loss=metrics.get("val_loss"),
                val_accuracy=metrics.get("val_accuracy"),
                num_samples=metrics.get("num_samples", 0),
                metrics=metrics
            )
            session.add(record)
            session.commit()
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        """Get experiment by ID."""
        with self.get_session() as session:
            experiment = session.query(Experiment).filter(
                Experiment.id == experiment_id
            ).first()
            
            if experiment:
                return {
                    "id": experiment.id,
                    "name": experiment.name,
                    "created_at": experiment.created_at.isoformat(),
                    "status": experiment.status,
                    "config": experiment.config,
                    "final_accuracy": experiment.final_accuracy,
                    "best_accuracy": experiment.best_accuracy
                }
            return None
    
    def get_experiments(self, limit: int = 100) -> List[Dict]:
        """Get all experiments."""
        with self.get_session() as session:
            experiments = session.query(Experiment).order_by(
                Experiment.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    "id": e.id,
                    "name": e.name,
                    "created_at": e.created_at.isoformat(),
                    "status": e.status,
                    "num_rounds": e.num_rounds,
                    "final_accuracy": e.final_accuracy,
                    "best_accuracy": e.best_accuracy
                }
                for e in experiments
            ]
    
    def get_round_metrics(
        self,
        experiment_id: int
    ) -> List[Dict]:
        """Get all round metrics for an experiment."""
        with self.get_session() as session:
            rounds = session.query(RoundRecord).filter(
                RoundRecord.experiment_id == experiment_id
            ).order_by(RoundRecord.round_num).all()
            
            return [
                {
                    "round_num": r.round_num,
                    "global_loss": r.global_loss,
                    "global_accuracy": r.global_accuracy,
                    "auc_roc": r.auc_roc,
                    "epsilon_spent": r.epsilon_spent,
                    "num_clients": r.num_clients,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in rounds
            ]


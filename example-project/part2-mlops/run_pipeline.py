"""
Main Entry Point: Run the Full ML Pipeline

This script executes all steps in order:
1. Data Ingestion
2. Preprocessing (SageMaker Processing Job)
3. Training (SageMaker Training Job)
4. Evaluation (SageMaker Processing Job)
5. Quality Gate Check
6. Model Registration (if passed)
7. Deployment (optional)
8. Monitoring Setup (optional)

Usage:
    uv run python run_pipeline.py
    uv run python run_pipeline.py --skip-deploy    # Skip deployment
    uv run python run_pipeline.py --deploy         # Include deployment
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import get_s3_paths, get_session, load_config
from src.data_ingestion import ingest
from src.deployment import deploy_model
from src.evaluation import check_quality_gates, evaluate_model
from src.monitoring import setup_cloudwatch_alarm, setup_model_monitor
from src.preprocessing import run_preprocessing
from src.registry import approve_model, register_model
from src.training import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main(deploy: bool = False, auto_approve: bool = False):
    """Run the complete ML pipeline."""
    logger.info("=" * 60)
    logger.info("   CUSTOMER CHURN PREDICTION - ML PIPELINE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config()
    session = get_session(config)
    s3_paths = get_s3_paths(config)

    logger.info(f"Project: {config['project_name']}")
    logger.info(f"Region: {config['region']}")
    logger.info(f"Bucket: {s3_paths['bucket']}")

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Data Ingestion
    # ─────────────────────────────────────────────────────────────
    raw_data_uri = ingest(session, s3_paths, n_samples=5000)

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Preprocessing
    # ─────────────────────────────────────────────────────────────
    processed_paths = run_preprocessing(
        session=session,
        role_arn=config['role_arn'],
        input_s3_uri=raw_data_uri,
        output_s3_base=s3_paths['processed'],
        instance_type=config['processing_instance_type'],
        scripts_dir='scripts'
    )

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Training
    # ─────────────────────────────────────────────────────────────
    estimator = train_model(
        session=session,
        role_arn=config['role_arn'],
        region=config['region'],
        train_s3_uri=processed_paths['train'],
        validation_s3_uri=processed_paths['validation'],
        output_s3_uri=s3_paths['models'],
        hyperparameters=config['hyperparameters'],
        instance_type=config['training_instance_type']
    )

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Evaluation
    # ─────────────────────────────────────────────────────────────
    eval_report_uri = evaluate_model(
        session=session,
        role_arn=config['role_arn'],
        model_data_s3=estimator.model_data,
        test_s3_uri=processed_paths['test'],
        output_s3_uri=s3_paths['evaluation'],
        instance_type=config['processing_instance_type'],
        scripts_dir='scripts'
    )

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Quality Gate
    # ─────────────────────────────────────────────────────────────
    passed = check_quality_gates(
        eval_report_s3=eval_report_uri,
        session=session,
        quality_gates=config['quality_gate']
    )

    if not passed:
        logger.error("Pipeline STOPPED: Quality gates failed.")
        logger.error("Action: Review metrics, improve features/model, retry.")
        return

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Model Registration
    # ─────────────────────────────────────────────────────────────
    model_arn = register_model(
        session=session,
        role_arn=config['role_arn'],
        region=config['region'],
        model_data_s3=estimator.model_data,
        model_package_group_name=config['model_package_group_name'],
        approval_status='Approved' if auto_approve else 'PendingManualApproval',
        description=f'Churn model - trained on {s3_paths["processed"]}'
    )

    # ─────────────────────────────────────────────────────────────
    # STEP 7: Deployment (optional)
    # ─────────────────────────────────────────────────────────────
    if deploy:
        endpoint_name = f"{config['project_name']}-endpoint"
        predictor = deploy_model(
            session=session,
            role_arn=config['role_arn'],
            region=config['region'],
            model_data_s3=estimator.model_data,
            endpoint_name=endpoint_name,
            instance_type=config.get('inference_instance_type', 'ml.m5.large')
        )

        # ─────────────────────────────────────────────────────────
        # STEP 8: Monitoring Setup
        # ─────────────────────────────────────────────────────────
        setup_cloudwatch_alarm(
            region=config['region'],
            endpoint_name=endpoint_name
        )
        logger.info(f"Endpoint live at: {endpoint_name}")
    else:
        logger.info("Deployment skipped (use --deploy to deploy)")

    # ─────────────────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("   ✅ PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Model artifact: {estimator.model_data}")
    logger.info(f"   Model registry: {model_arn}")
    if deploy:
        logger.info(f"   Endpoint: {endpoint_name}")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ML Pipeline')
    parser.add_argument('--deploy', action='store_true', help='Deploy model to endpoint')
    parser.add_argument('--auto-approve', action='store_true', help='Auto-approve model')
    args = parser.parse_args()

    main(deploy=args.deploy, auto_approve=args.auto_approve)

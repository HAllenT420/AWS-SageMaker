"""
Module 8: Model Monitoring

Responsible for:
- Setting up SageMaker Model Monitor
- Creating baselines from training data
- Scheduling monitoring jobs
- Configuring drift alerts
"""

import logging

import boto3
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

logger = logging.getLogger(__name__)


def setup_model_monitor(
    session,
    role_arn: str,
    endpoint_name: str,
    baseline_data_s3: str,
    output_s3_uri: str,
    schedule_expression: str = 'cron(0 */6 ? * * *)'  # Every 6 hours
):
    """
    Set up Model Monitor for an endpoint.

    What this does:
    1. Creates a baseline from your training data (expected distributions)
    2. Schedules periodic checks against live traffic
    3. Flags when distributions drift beyond thresholds

    Args:
        session: SageMaker session
        role_arn: IAM role
        endpoint_name: Endpoint to monitor
        baseline_data_s3: S3 path to training data (for baseline)
        output_s3_uri: S3 path for monitoring outputs
        schedule_expression: How often to check (cron format)
    """
    logger.info("=" * 50)
    logger.info("STEP 8: MODEL MONITORING SETUP")
    logger.info("=" * 50)

    monitor = DefaultModelMonitor(
        role=role_arn,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=session
    )

    # Step 1: Create baseline (what "normal" looks like)
    logger.info("Creating baseline from training data...")
    monitor.suggest_baseline(
        baseline_dataset=baseline_data_s3,
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=f'{output_s3_uri}/baseline'
    )
    logger.info("Baseline created.")

    # Step 2: Schedule monitoring
    logger.info(f"Scheduling monitor: {schedule_expression}")
    monitor.create_monitoring_schedule(
        monitor_schedule_name=f'{endpoint_name}-monitor',
        endpoint_input=endpoint_name,
        output_s3_uri=f'{output_s3_uri}/results',
        schedule_cron_expression=schedule_expression
    )

    logger.info(f"✅ Monitor scheduled for endpoint: {endpoint_name}")
    return monitor


def setup_cloudwatch_alarm(
    region: str,
    endpoint_name: str,
    alarm_name: str = None,
    sns_topic_arn: str = None
):
    """
    Set up CloudWatch alarm for endpoint errors.

    Args:
        region: AWS region
        endpoint_name: Endpoint to monitor
        alarm_name: Name for the alarm
        sns_topic_arn: SNS topic for notifications
    """
    if alarm_name is None:
        alarm_name = f'{endpoint_name}-high-error-rate'

    cloudwatch = boto3.client('cloudwatch', region_name=region)

    alarm_config = {
        'AlarmName': alarm_name,
        'MetricName': 'Invocation5XXErrors',
        'Namespace': 'AWS/SageMaker',
        'Statistic': 'Sum',
        'Period': 300,  # 5 minutes
        'EvaluationPeriods': 2,
        'Threshold': 5,
        'ComparisonOperator': 'GreaterThanThreshold',
        'Dimensions': [
            {'Name': 'EndpointName', 'Value': endpoint_name},
            {'Name': 'VariantName', 'Value': 'AllTraffic'}
        ],
        'TreatMissingData': 'notBreaching'
    }

    if sns_topic_arn:
        alarm_config['AlarmActions'] = [sns_topic_arn]

    cloudwatch.put_metric_alarm(**alarm_config)
    logger.info(f"✅ CloudWatch alarm created: {alarm_name}")

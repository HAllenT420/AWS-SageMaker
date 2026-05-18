"""
SageMaker Pipeline Definition (Fully Automated)

This defines the pipeline as a SageMaker Pipeline object that can be:
- Triggered on schedule (EventBridge)
- Triggered by drift detection
- Triggered by CI/CD

This is the "final form" of your ML pipeline — fully managed by AWS.

Usage:
    uv run python pipeline/pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from config.config import get_s3_paths, get_session, load_config


def create_pipeline():
    """Create the SageMaker Pipeline."""
    config = load_config()
    session = get_session(config)
    s3_paths = get_s3_paths(config)
    role = config['role_arn']
    region = config['region']

    scripts_dir = str(Path(__file__).parent.parent / 'scripts')

    # === PARAMETERS (overridable at runtime) ===
    param_input_data = ParameterString(
        name='InputDataUrl',
        default_value=f"{s3_paths['raw_data']}/raw_data.csv"
    )
    param_auc_threshold = ParameterFloat(name='AucThreshold', default_value=0.75)

    # === STEP 1: PREPROCESS ===
    processor = SKLearnProcessor(
        framework_version='1.2-1',
        role=role,
        instance_type=config['processing_instance_type'],
        instance_count=1,
        sagemaker_session=session,
        base_job_name='churn-preprocess'
    )

    step_preprocess = ProcessingStep(
        name='Preprocess',
        processor=processor,
        code=f'{scripts_dir}/preprocess.py',
        inputs=[ProcessingInput(source=param_input_data, destination='/opt/ml/processing/input')],
        outputs=[
            ProcessingOutput(output_name='train', source='/opt/ml/processing/output/train'),
            ProcessingOutput(output_name='validation', source='/opt/ml/processing/output/validation'),
            ProcessingOutput(output_name='test', source='/opt/ml/processing/output/test'),
        ]
    )

    # === STEP 2: TRAIN ===
    xgb_image = sagemaker.image_uris.retrieve('xgboost', region, '1.7-1')
    estimator = Estimator(
        image_uri=xgb_image,
        role=role,
        instance_count=1,
        instance_type=config['training_instance_type'],
        output_path=s3_paths['models'],
        sagemaker_session=session
    )
    estimator.set_hyperparameters(**config['hyperparameters'])

    step_train = TrainingStep(
        name='Train',
        estimator=estimator,
        inputs={
            'train': TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri,
                content_type='text/csv'
            ),
            'validation': TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs['validation'].S3Output.S3Uri,
                content_type='text/csv'
            )
        }
    )

    # === STEP 3: EVALUATE ===
    eval_report = PropertyFile(name='EvalReport', output_name='evaluation', path='evaluation.json')

    step_evaluate = ProcessingStep(
        name='Evaluate',
        processor=processor,
        code=f'{scripts_dir}/evaluate.py',
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination='/opt/ml/processing/model'
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs['test'].S3Output.S3Uri,
                destination='/opt/ml/processing/test'
            )
        ],
        outputs=[ProcessingOutput(output_name='evaluation', source='/opt/ml/processing/evaluation')],
        property_files=[eval_report]
    )

    # === STEP 4: CONDITIONAL REGISTRATION ===
    step_register = RegisterModel(
        name='RegisterModel',
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=['text/csv'],
        response_types=['text/csv'],
        inference_instances=['ml.m5.large'],
        transform_instances=['ml.m5.large'],
        model_package_group_name=config['model_package_group_name'],
        approval_status='PendingManualApproval'
    )

    condition = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate.name,
            property_file=eval_report,
            json_path='binary_classification_metrics.auc.value'
        ),
        right=param_auc_threshold
    )

    step_condition = ConditionStep(
        name='CheckAUC',
        conditions=[condition],
        if_steps=[step_register],
        else_steps=[]
    )

    # === ASSEMBLE PIPELINE ===
    pipeline = Pipeline(
        name=config['pipeline_name'],
        parameters=[param_input_data, param_auc_threshold],
        steps=[step_preprocess, step_train, step_evaluate, step_condition],
        sagemaker_session=session
    )

    return pipeline


if __name__ == '__main__':
    config = load_config()
    print(f"Creating pipeline: {config['pipeline_name']}")

    pipeline = create_pipeline()
    pipeline.upsert(role_arn=config['role_arn'])
    print("Pipeline upserted.")

    execution = pipeline.start()
    print(f"Execution started: {execution.arn}")
    print(f"\nMonitor at: https://{config['region']}.console.aws.amazon.com/sagemaker/home#/pipelines")

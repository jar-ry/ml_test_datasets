from de_pipeline.nodes.preprocess_data import preprocess_data

def run_pipeline(is_local):
    """
        Orchestrate and run ML pipeline
    """
    preprocess_data(
        input_data=["housing"],
        output_data=["processed_housing"],
        is_local=is_local
    )
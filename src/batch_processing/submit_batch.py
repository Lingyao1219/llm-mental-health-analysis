"""
Submit prepared batch file to OpenAI Batch API.
Monitor the batch job status.
"""

import time
import json
import logging
from pathlib import Path
from openai import OpenAI

import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")


def submit_batch_job(client: OpenAI, input_file_path: str, base_name: str):
    """Upload batch input file and create a batch job."""
    logging.info(f"Uploading batch file: {input_file_path}")

    # Upload the file
    with open(input_file_path, 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )

    logging.info(f"✅ File uploaded with ID: {batch_input_file.id}")

    # Create batch job
    logging.info("Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"LLM mental health impact extraction - {base_name}",
            "source_file": base_name
        }
    )

    logging.info(f"✅ Batch job created with ID: {batch_job.id}")
    logging.info(f"   Status: {batch_job.status}")
    logging.info(f"   Total requests: {batch_job.request_counts.total}")

    # Save batch info to batch folder with proper naming
    batch_folder = Path(cfg.BATCH_FOLDER)
    batch_folder.mkdir(exist_ok=True)

    batch_info_file = batch_folder / f"{base_name}_batch_info.json"
    batch_info = {
        "base_name": base_name,
        "batch_id": batch_job.id,
        "input_file_id": batch_input_file.id,
        "status": batch_job.status,
        "created_at": batch_job.created_at
    }

    with open(batch_info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)

    logging.info(f"✅ Batch info saved to: {batch_info_file}")

    return batch_job


def check_batch_status(client: OpenAI, batch_id: str, batch_info_file: Path):
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)

    logging.info(f"\nBatch Status: {batch.status}")
    logging.info(f"Request counts:")
    logging.info(f"  Total: {batch.request_counts.total}")
    logging.info(f"  Completed: {batch.request_counts.completed}")
    logging.info(f"  Failed: {batch.request_counts.failed}")

    if batch.status == "completed":
        logging.info(f"✅ Batch completed!")
        logging.info(f"   Output file ID: {batch.output_file_id}")

        # Load existing batch info to preserve base_name
        with open(batch_info_file, 'r') as f:
            batch_info = json.load(f)

        # Update with completion info
        batch_info.update({
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "completed_at": batch.completed_at
        })

        with open(batch_info_file, 'w') as f:
            json.dump(batch_info, f, indent=2)

        return batch

    elif batch.status == "failed":
        logging.error(f"❌ Batch failed!")
        if batch.errors:
            logging.error(f"Errors: {batch.errors}")

    return batch


def download_results(client: OpenAI, batch: any, base_name: str):
    """Download batch results."""
    if batch.status != "completed":
        logging.warning("Batch not completed yet. Cannot download results.")
        return None

    # Check if output_file_id exists
    if not batch.output_file_id:
        logging.error(f"❌ Batch completed but no output file ID found!")
        logging.error(f"   Batch status: {batch.status}")
        logging.error(f"   Request counts - Total: {batch.request_counts.total}, Completed: {batch.request_counts.completed}, Failed: {batch.request_counts.failed}")
        if batch.errors:
            logging.error(f"   Batch errors: {batch.errors}")
        return None

    logging.info(f"Downloading results from file ID: {batch.output_file_id}")

    # Create batch folder if needed
    batch_folder = Path(cfg.BATCH_FOLDER)
    batch_folder.mkdir(exist_ok=True)

    # Download output file with proper naming: sample_batch_output.jsonl
    result_file = client.files.content(batch.output_file_id)
    result_path = batch_folder / f"{base_name}_batch_output.jsonl"

    with open(result_path, 'wb') as f:
        f.write(result_file.content)

    logging.info(f"✅ Results downloaded to: {result_path}")

    # Download error file if exists
    if batch.error_file_id:
        logging.info(f"Downloading errors from file ID: {batch.error_file_id}")
        error_file = client.files.content(batch.error_file_id)
        error_path = batch_folder / f"{base_name}_batch_errors.jsonl"

        with open(error_path, 'wb') as f:
            f.write(error_file.content)

        logging.info(f"✅ Errors downloaded to: {error_path}")

    return result_path


def monitor_batch(client: OpenAI, batch_id: str, batch_info_file: Path, check_interval: int = 60):
    """Monitor batch job until completion."""
    logging.info(f"Monitoring batch {batch_id}...")
    logging.info(f"Checking every {check_interval} seconds...")
    logging.info("(Press Ctrl+C to stop monitoring. You can resume later.)\n")

    # Load base_name from batch info
    with open(batch_info_file, 'r') as f:
        batch_info = json.load(f)
    base_name = batch_info.get('base_name', 'unknown')

    try:
        while True:
            batch = check_batch_status(client, batch_id, batch_info_file)

            if batch.status in ["completed", "failed", "expired", "cancelled"]:
                if batch.status == "completed":
                    download_results(client, batch, base_name)
                break

            logging.info(f"Waiting {check_interval} seconds before next check...\n")
            time.sleep(check_interval)

    except KeyboardInterrupt:
        logging.warning("\n🛑 Monitoring stopped by user.")
        logging.info(f"You can resume monitoring by running:")
        logging.info(f"  python submit_batch.py --check {batch_id}")


def main():
    """Main function."""
    import sys

    logging.info("=" * 80)
    logging.info("OpenAI Batch API Submission")
    logging.info("=" * 80)

    # Initialize client
    client = OpenAI(api_key=cfg.API_KEY)

    # Determine the batch input file
    batch_folder = Path(cfg.BATCH_FOLDER)

    # Find batch input files
    batch_input_files = list(batch_folder.glob("*_batch_input.jsonl"))

    if not batch_input_files:
        logging.error(f"No batch input files found in {batch_folder}")
        logging.error("Please run prepare_batch.py first!")
        sys.exit(1)

    # Use the most recent batch input file
    input_file_path = sorted(batch_input_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    base_name = input_file_path.stem.replace("_batch_input", "")  # "sample_batch_input" -> "sample"

    logging.info(f"Using batch input file: {input_file_path}")
    logging.info(f"Base name: {base_name}")

    # Check if batch info exists for this file
    batch_info_file = batch_folder / f"{base_name}_batch_info.json"

    if batch_info_file.exists():
        # Resume existing batch
        with open(batch_info_file, 'r') as f:
            batch_info = json.load(f)

        batch_id = batch_info['batch_id']
        logging.info(f"Found existing batch: {batch_id}")
        logging.info(f"Resuming monitoring...")

        monitor_batch(client, batch_id, batch_info_file)

    else:
        # Submit new batch
        batch_job = submit_batch_job(client, str(input_file_path), base_name)

        logging.info("\n" + "=" * 80)
        logging.info("Batch submitted successfully!")
        logging.info(f"Batch ID: {batch_job.id}")
        logging.info("\nStarting monitoring...")
        logging.info("=" * 80 + "\n")

        monitor_batch(client, batch_job.id, batch_info_file)


if __name__ == "__main__":
    main()

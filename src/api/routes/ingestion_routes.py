"""
Document Ingestion API Routes
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import asyncio
from loguru import logger

from src.knowledge.ingestion.pipeline import IngestionPipeline
from src.knowledge.ingestion.arxiv_fetcher import ArxivFetcher
from src.api.websocket_manager import websocket_manager


router = APIRouter()


class ArxivIngestionRequest(BaseModel):
    """Request model for arXiv paper ingestion"""
    query: Optional[str] = None
    arxiv_ids: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    max_papers: int = Field(default=10, le=100)
    categories: Optional[List[str]] = None
    recent_only: bool = Field(default=True)


class PDFIngestionRequest(BaseModel):
    """Request model for PDF ingestion"""
    file_paths: List[str] = Field(..., description="List of PDF file paths")
    extract_formulas: bool = Field(default=True)
    extract_tables: bool = Field(default=True)
    use_ocr: bool = Field(default=False)


class IngestionStatus(BaseModel):
    """Ingestion status response"""
    task_id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    errors: List[str]
    created_at: datetime
    completed_at: Optional[datetime] = None


# Global ingestion tasks tracker
ingestion_tasks = {}


@router.post("/arxiv")
async def ingest_arxiv_papers(
    request: ArxivIngestionRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Ingest papers from arXiv

    Args:
        request: ArXiv ingestion request
        background_tasks: Background tasks
        req: Request object

    Returns:
        Ingestion task information
    """
    try:
        # Create task ID
        task_id = f"arxiv_{datetime.now().timestamp()}"

        # Initialize task status
        ingestion_tasks[task_id] = IngestionStatus(
            task_id=task_id,
            status="started",
            total_documents=0,
            processed_documents=0,
            failed_documents=0,
            errors=[],
            created_at=datetime.now()
        )

        # Broadcast start
        await websocket_manager.broadcast_task_update(
            task_id,
            "started",
            0,
            {"type": "arxiv_ingestion"}
        )

        # Run ingestion in background
        async def run_arxiv_ingestion():
            try:
                fetcher = ArxivFetcher()
                pipeline = IngestionPipeline()

                papers = []

                # Fetch papers based on request type
                if request.arxiv_ids:
                    for arxiv_id in request.arxiv_ids:
                        paper = await fetcher.fetch_paper_by_id(arxiv_id)
                        if paper:
                            papers.append(paper)
                elif request.query:
                    papers = await fetcher.search_papers(
                        query=request.query,
                        max_results=request.max_papers,
                        categories=request.categories
                    )
                elif request.topics:
                    papers = await fetcher.fetch_mathematical_papers(
                        topics=request.topics,
                        max_papers=request.max_papers,
                        recent_only=request.recent_only
                    )
                else:
                    # Default: fetch recent mathematical papers
                    papers = await fetcher.fetch_mathematical_papers(
                        max_papers=request.max_papers,
                        recent_only=request.recent_only
                    )

                # Update total documents
                ingestion_tasks[task_id].total_documents = len(papers)

                # Download and process papers
                for i, paper in enumerate(papers):
                    try:
                        # Download PDF
                        paths = await fetcher.download_paper(paper, download_pdf=True)
                        paper["local_paths"] = paths

                        # Process with pipeline
                        if "pdf" in paths:
                            result = await pipeline.process_document(
                                document_path=paths["pdf"],
                                document_type="pdf",
                                metadata=paper
                            )

                            if result["success"]:
                                ingestion_tasks[task_id].processed_documents += 1
                            else:
                                ingestion_tasks[task_id].failed_documents += 1
                                ingestion_tasks[task_id].errors.append(
                                    f"Failed to process {paper['title']}: {result.get('error')}"
                                )

                        # Broadcast progress
                        progress = ((i + 1) / len(papers)) * 100
                        await websocket_manager.broadcast_task_update(
                            task_id,
                            "processing",
                            progress,
                            {
                                "current_paper": paper["title"],
                                "processed": i + 1,
                                "total": len(papers)
                            }
                        )

                    except Exception as e:
                        logger.error(f"Error processing paper {paper.get('title')}: {e}")
                        ingestion_tasks[task_id].failed_documents += 1
                        ingestion_tasks[task_id].errors.append(str(e))

                # Complete task
                ingestion_tasks[task_id].status = "completed"
                ingestion_tasks[task_id].completed_at = datetime.now()

                await websocket_manager.broadcast_task_update(
                    task_id,
                    "completed",
                    100,
                    {
                        "processed": ingestion_tasks[task_id].processed_documents,
                        "failed": ingestion_tasks[task_id].failed_documents
                    }
                )

            except Exception as e:
                logger.error(f"ArXiv ingestion failed: {e}")
                ingestion_tasks[task_id].status = "failed"
                ingestion_tasks[task_id].errors.append(str(e))

                await websocket_manager.broadcast_task_update(
                    task_id,
                    "failed",
                    0,
                    {"error": str(e)}
                )

        background_tasks.add_task(run_arxiv_ingestion)

        return ingestion_tasks[task_id]

    except Exception as e:
        logger.error(f"Error starting arXiv ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pdf")
async def ingest_pdf_documents(
    request: PDFIngestionRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Ingest PDF documents

    Args:
        request: PDF ingestion request
        background_tasks: Background tasks
        req: Request object

    Returns:
        Ingestion task information
    """
    try:
        # Create task ID
        task_id = f"pdf_{datetime.now().timestamp()}"

        # Initialize task status
        ingestion_tasks[task_id] = IngestionStatus(
            task_id=task_id,
            status="started",
            total_documents=len(request.file_paths),
            processed_documents=0,
            failed_documents=0,
            errors=[],
            created_at=datetime.now()
        )

        # Broadcast start
        await websocket_manager.broadcast_task_update(
            task_id,
            "started",
            0,
            {"type": "pdf_ingestion", "total_files": len(request.file_paths)}
        )

        # Run ingestion in background
        async def run_pdf_ingestion():
            try:
                pipeline = IngestionPipeline()

                for i, file_path in enumerate(request.file_paths):
                    try:
                        path = Path(file_path)
                        if not path.exists():
                            raise FileNotFoundError(f"File not found: {file_path}")

                        # Process with pipeline
                        result = await pipeline.process_document(
                            document_path=path,
                            document_type="pdf",
                            metadata={
                                "source": "local",
                                "extract_formulas": request.extract_formulas,
                                "extract_tables": request.extract_tables,
                                "use_ocr": request.use_ocr
                            }
                        )

                        if result["success"]:
                            ingestion_tasks[task_id].processed_documents += 1
                        else:
                            ingestion_tasks[task_id].failed_documents += 1
                            ingestion_tasks[task_id].errors.append(
                                f"Failed to process {path.name}: {result.get('error')}"
                            )

                        # Broadcast progress
                        progress = ((i + 1) / len(request.file_paths)) * 100
                        await websocket_manager.broadcast_task_update(
                            task_id,
                            "processing",
                            progress,
                            {
                                "current_file": path.name,
                                "processed": i + 1,
                                "total": len(request.file_paths)
                            }
                        )

                    except Exception as e:
                        logger.error(f"Error processing PDF {file_path}: {e}")
                        ingestion_tasks[task_id].failed_documents += 1
                        ingestion_tasks[task_id].errors.append(str(e))

                # Complete task
                ingestion_tasks[task_id].status = "completed"
                ingestion_tasks[task_id].completed_at = datetime.now()

                await websocket_manager.broadcast_task_update(
                    task_id,
                    "completed",
                    100,
                    {
                        "processed": ingestion_tasks[task_id].processed_documents,
                        "failed": ingestion_tasks[task_id].failed_documents
                    }
                )

            except Exception as e:
                logger.error(f"PDF ingestion failed: {e}")
                ingestion_tasks[task_id].status = "failed"
                ingestion_tasks[task_id].errors.append(str(e))

                await websocket_manager.broadcast_task_update(
                    task_id,
                    "failed",
                    0,
                    {"error": str(e)}
                )

        background_tasks.add_task(run_pdf_ingestion)

        return ingestion_tasks[task_id]

    except Exception as e:
        logger.error(f"Error starting PDF ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_and_ingest(
    background_tasks: BackgroundTasks,
    req: Request,
    files: List[UploadFile] = File(...)
):
    """
    Upload and ingest PDF files

    Args:
        background_tasks: Background tasks
        req: Request object
        files: List of uploaded files

    Returns:
        Ingestion task information
    """
    try:
        # Create task ID
        task_id = f"upload_{datetime.now().timestamp()}"

        # Save uploaded files
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)

        saved_paths = []
        for file in files:
            if file.filename.lower().endswith('.pdf'):
                file_path = upload_dir / file.filename
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                saved_paths.append(str(file_path))

        if not saved_paths:
            raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

        # Create ingestion request
        request = PDFIngestionRequest(
            file_paths=saved_paths,
            extract_formulas=True,
            extract_tables=True
        )

        # Process with PDF ingestion
        return await ingest_pdf_documents(request, background_tasks, req)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def get_ingestion_status(task_id: str):
    """
    Get ingestion task status

    Args:
        task_id: Task identifier

    Returns:
        Task status information
    """
    if task_id not in ingestion_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return ingestion_tasks[task_id]


@router.get("/tasks")
async def list_ingestion_tasks(
    status: Optional[str] = None,
    limit: int = 10
):
    """
    List ingestion tasks

    Args:
        status: Filter by status
        limit: Maximum number of tasks

    Returns:
        List of ingestion tasks
    """
    tasks = list(ingestion_tasks.values())

    if status:
        tasks = [t for t in tasks if t.status == status]

    # Sort by creation time (newest first)
    tasks.sort(key=lambda t: t.created_at, reverse=True)

    return tasks[:limit]


@router.delete("/tasks/{task_id}")
async def cancel_ingestion_task(task_id: str):
    """
    Cancel an ingestion task

    Args:
        task_id: Task identifier

    Returns:
        Cancellation confirmation
    """
    if task_id not in ingestion_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = ingestion_tasks[task_id]

    if task.status in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed task")

    task.status = "cancelled"
    task.completed_at = datetime.now()

    await websocket_manager.broadcast_task_update(
        task_id,
        "cancelled",
        0,
        {"message": "Task cancelled by user"}
    )

    return {"status": "success", "message": f"Task {task_id} cancelled"}


@router.post("/batch")
async def batch_ingest(
    background_tasks: BackgroundTasks,
    req: Request,
    arxiv_request: Optional[ArxivIngestionRequest] = None,
    pdf_request: Optional[PDFIngestionRequest] = None
):
    """
    Batch ingestion of multiple sources

    Args:
        background_tasks: Background tasks
        req: Request object
        arxiv_request: ArXiv ingestion request
        pdf_request: PDF ingestion request

    Returns:
        Batch task information
    """
    try:
        tasks = []

        if arxiv_request:
            arxiv_task = await ingest_arxiv_papers(arxiv_request, background_tasks, req)
            tasks.append(arxiv_task)

        if pdf_request:
            pdf_task = await ingest_pdf_documents(pdf_request, background_tasks, req)
            tasks.append(pdf_task)

        if not tasks:
            raise HTTPException(status_code=400, detail="No ingestion requests provided")

        return {
            "batch_id": f"batch_{datetime.now().timestamp()}",
            "tasks": tasks,
            "total_tasks": len(tasks)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_ingestion_statistics():
    """
    Get ingestion statistics

    Returns:
        Ingestion statistics
    """
    total_tasks = len(ingestion_tasks)
    completed_tasks = sum(1 for t in ingestion_tasks.values() if t.status == "completed")
    failed_tasks = sum(1 for t in ingestion_tasks.values() if t.status == "failed")
    active_tasks = sum(1 for t in ingestion_tasks.values() if t.status in ["started", "processing"])

    total_documents = sum(t.total_documents for t in ingestion_tasks.values())
    processed_documents = sum(t.processed_documents for t in ingestion_tasks.values())
    failed_documents = sum(t.failed_documents for t in ingestion_tasks.values())

    return {
        "tasks": {
            "total": total_tasks,
            "completed": completed_tasks,
            "failed": failed_tasks,
            "active": active_tasks
        },
        "documents": {
            "total": total_documents,
            "processed": processed_documents,
            "failed": failed_documents,
            "success_rate": (processed_documents / total_documents * 100) if total_documents > 0 else 0
        }
    }


@router.delete("/tasks")
async def clear_completed_tasks():
    """
    Clear completed ingestion tasks

    Returns:
        Number of tasks cleared
    """
    completed_tasks = [
        task_id for task_id, task in ingestion_tasks.items()
        if task.status in ["completed", "failed", "cancelled"]
    ]

    for task_id in completed_tasks:
        del ingestion_tasks[task_id]

    return {
        "status": "success",
        "cleared": len(completed_tasks),
        "message": f"Cleared {len(completed_tasks)} completed tasks"
    }
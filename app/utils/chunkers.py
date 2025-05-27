import logging
from typing import Any, Dict, Generator, List, Optional

import tiktoken
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean_extra_whitespace, clean_non_ascii_chars
from unstructured.documents.elements import CompositeElement, Element, Text
from unstructured.partition.auto import partition

logger = logging.getLogger("cx_consulting_ai.chunkers")

# Constants for chunking
MAX_CHUNK_SIZE_TOKENS = 500
# Overlap for splitting oversized elements (character count for simplicity here)
# Using a fraction of the target token size as a heuristic for char overlap
# Adjust char overlap calculation based on typical token:char ratio if needed.
CHAR_OVERLAP = int(MAX_CHUNK_SIZE_TOKENS * 0.2 * 4)  # Approx 20% overlap in chars

# Get the encoding directly
try:
    ENCODING = tiktoken.get_encoding("gpt2")
except Exception as e:
    logger.error(f"Failed to get tiktoken gpt2 encoding: {e}. Using fallback counting.")
    ENCODING = None


# Define a helper function to get token count using the specific encoding
def get_token_count(text: str) -> int:
    if ENCODING:
        try:
            return len(ENCODING.encode(text))
        except Exception as e:
            logger.warning(
                f"Tiktoken encoding failed: {e}. Falling back to char count / 4."
            )
            return len(text) // 4  # Fallback
    else:
        return len(text) // 4  # Fallback if encoding failed to load


def _split_text_with_overlap(
    text: str, max_len_chars: int, overlap_chars: int
) -> Generator[str, None, None]:
    """Helper to split long text by characters with overlap."""
    if len(text) <= max_len_chars:
        yield text
        return

    start = 0
    while start < len(text):
        end = min(start + max_len_chars, len(text))
        yield text[start:end]
        if end == len(text):
            break
        start += max_len_chars - overlap_chars
        # Ensure start doesn't go backward in case of very large overlap/small max_len
        start = max(start, end - max_len_chars + 1)


def chunk_document(
    file_path: str,
    file_type: Optional[str] = None,
    doc_metadata: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Chunks a document using unstructured partition, cleans, merges elements,
    and splits oversized single elements with overlap.

    Args:
        file_path: Path to the document file.
        file_type: Explicit file type (e.g., 'pdf', 'docx'). If None, unstructured tries to detect.
        doc_metadata: Base metadata associated with the whole document.

    Returns:
        A list of dictionaries, where each dictionary represents a chunk
        with 'text' and 'metadata' keys. Metadata includes page numbers,
        source type, and potentially headings.
    """
    if not doc_metadata:
        doc_metadata = {}

    logger.info(
        f"Starting chunking for file: {file_path} (type: {file_type or 'auto'})"
    )

    try:
        # Use unstructured partition function - handles various types
        # strategy="hi_res" can be used for better PDF layout analysis if needed, requires Detectron2
        elements: List[Element] = partition(
            filename=file_path, content_type=file_type, strategy="fast"
        )

        # Optional: Clean elements (remove extra whitespace, etc.)
        for element in elements:
            if isinstance(element, Text):
                element.text = clean_extra_whitespace(element.text)
                # element.text = clean_non_ascii_chars(element.text) # Optional: might remove useful chars

    except Exception as e:
        logger.error(
            f"Unstructured partitioning failed for {file_path}: {e}", exc_info=True
        )
        return []

    if not elements:
        logger.warning(
            f"Unstructured partitioning returned no elements for {file_path}"
        )
        return []

    # --- Chunking Strategy: Merge elements & split oversized ones ---
    merged_chunks = []
    current_chunk_text = ""
    current_chunk_meta = {}
    current_chunk_index = 0  # ADDED: To track chunk index for ID generation
    first_element_in_chunk = True
    # Estimate max characters based on token limit (heuristic)
    max_chars_approx = MAX_CHUNK_SIZE_TOKENS * 5  # Assuming avg 5 chars/token

    for i, element in enumerate(elements):
        # Skip empty elements
        if not isinstance(element, Text) or not element.text.strip():
            continue

        element_text = element.text  # Use cleaned text
        element_meta = element.metadata.to_dict()
        element_token_count = get_token_count(element_text)

        # Can we add this element to the current chunk?
        separator = "\n\n" if current_chunk_text else ""
        potential_new_text = current_chunk_text + separator + element_text
        potential_token_count = get_token_count(potential_new_text)

        # Condition: Fits, and not the very last element (to ensure last one is processed)
        if potential_token_count <= MAX_CHUNK_SIZE_TOKENS and i < len(elements) - 1:
            current_chunk_text = potential_new_text
            if first_element_in_chunk:
                # Base metadata for chunk
                current_chunk_meta = {
                    **doc_metadata,
                    "page_number": element_meta.get("page_number"),
                    "filename": element_meta.get("filename"),
                    # ADDED chunk_id (Checklist Item 5-A)
                    "chunk_id": f"{doc_metadata.get('doc_id', 'unknown')}_{current_chunk_index}",
                }
                if hasattr(element, "category") and element.category == "Title":
                    current_chunk_meta["heading"] = element_text[:100]
                first_element_in_chunk = False
            # Update end page number
            if element_meta.get("page_number"):
                current_chunk_meta["end_page_number"] = element_meta.get("page_number")
        else:
            # Finish the PREVIOUS chunk (if any text accumulated)
            if current_chunk_text:
                current_chunk_meta["text_token_count"] = get_token_count(
                    current_chunk_text
                )
                current_chunk_meta["source_type"] = doc_metadata.get("document_type")
                current_chunk_meta["template_name"] = doc_metadata.get("template_name")
                # Ensure chunk_id exists before appending
                if "chunk_id" not in current_chunk_meta:
                    current_chunk_meta["chunk_id"] = (
                        f"{doc_metadata.get('doc_id', 'unknown')}_{current_chunk_index}"
                    )

                merged_chunks.append(
                    {
                        "text": current_chunk_text.strip(),
                        "metadata": current_chunk_meta,
                    }
                )
                current_chunk_index += (
                    1  # Increment chunk index AFTER finishing a chunk
                )
                current_chunk_text = ""  # Reset for next potential chunk
                first_element_in_chunk = True

            # Now process the CURRENT element (which either didn't fit or is the last one)

            # Check if this single element is oversized
            if element_token_count > MAX_CHUNK_SIZE_TOKENS:
                logger.warning(
                    f"Element {i} from {element_meta.get('filename')} exceeds token limit ({element_token_count} > {MAX_CHUNK_SIZE_TOKENS}). Splitting with overlap."
                )
                # Split the oversized element text with character overlap
                sub_chunks = _split_text_with_overlap(
                    element_text, max_chars_approx, CHAR_OVERLAP
                )
                for sub_chunk_text in sub_chunks:
                    sub_chunk_token_count = get_token_count(sub_chunk_text)
                    if sub_chunk_token_count == 0:
                        continue  # Skip empty sub-chunks

                    # Create metadata for the sub-chunk, inheriting element/doc meta
                    sub_chunk_meta = {
                        **doc_metadata,
                        "page_number": element_meta.get(
                            "page_number"
                        ),  # Retain page number if available
                        "filename": element_meta.get("filename"),
                        "heading": current_chunk_meta.get("heading")
                        or (
                            element_text[:100]
                            if hasattr(element, "category")
                            and element.category == "Title"
                            else None
                        ),
                        "text_token_count": sub_chunk_token_count,
                        "source_type": doc_metadata.get("document_type"),
                        "template_name": doc_metadata.get("template_name"),
                        "part_of_split": True,  # Indicate it came from a split
                        # ADDED chunk_id for split chunks (Checklist Item 5-A)
                        "chunk_id": f"{doc_metadata.get('doc_id', 'unknown')}_{current_chunk_index}",
                    }
                    merged_chunks.append(
                        {
                            "text": sub_chunk_text.strip(),
                            "metadata": sub_chunk_meta,
                        }
                    )
                    current_chunk_index += 1  # Increment index for each sub-chunk
            else:
                # Element fits within limit or is the last one, start new chunk with it
                current_chunk_text = element_text
                current_chunk_meta = {  # Reset metadata for new chunk
                    **doc_metadata,
                    "page_number": element_meta.get("page_number"),
                    "filename": element_meta.get("filename"),
                    # ADDED chunk_id (Checklist Item 5-A)
                    "chunk_id": f"{doc_metadata.get('doc_id', 'unknown')}_{current_chunk_index}",
                }
                if hasattr(element, "category") and element.category == "Title":
                    current_chunk_meta["heading"] = element_text[:100]
                if element_meta.get("page_number"):
                    current_chunk_meta["end_page_number"] = element_meta.get(
                        "page_number"
                    )
                first_element_in_chunk = False  # Mark as not the first element anymore

                # If this IS the last element, finish this chunk immediately
                if i == len(elements) - 1:
                    current_chunk_meta["text_token_count"] = get_token_count(
                        current_chunk_text
                    )
                    current_chunk_meta["source_type"] = doc_metadata.get(
                        "document_type"
                    )
                    current_chunk_meta["template_name"] = doc_metadata.get(
                        "template_name"
                    )
                    merged_chunks.append(
                        {
                            "text": current_chunk_text.strip(),
                            "metadata": current_chunk_meta,
                        }
                    )
                    current_chunk_index += (
                        1  # Increment chunk index AFTER finishing the last chunk
                    )

    # Filter out potentially empty chunks from final list
    final_chunks = [chunk for chunk in merged_chunks if chunk.get("text")]

    logger.info(
        f"Generated {len(final_chunks)} chunks from {file_path} using unstructured, merging, and splitting."
    )
    return final_chunks


# Example usage (optional, for testing)
# if __name__ == '__main__':
#     # Create a dummy file for testing
#     dummy_file = "dummy_doc.txt"
#     with open(dummy_file, "w") as f:
#         f.write("This is the first sentence.\n\nThis is a second paragraph.\nIt has two lines.")
#
#     base_meta = {"document_type": "txt", "source": "local_test"}
#     chunks = chunk_document(dummy_file, file_type='txt', doc_metadata=base_meta)
#     for i, chunk in enumerate(chunks):
#         print(f"--- Chunk {i+1} ---")
#         print(f"Metadata: {chunk['metadata']}")
#         print(f"Text: {chunk['text']}")
#         print("-------------")
#
#     # Clean up dummy file
#     import os
#     os.remove(dummy_file)

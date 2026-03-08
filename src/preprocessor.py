"""
Handles raw data cleaning, text normalisation, and overlap-based chunking.
Stripping out headers, quoted lines, signatures, and junk metadata.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List


class NewsGroupPreprocessor:
    """
    Data preprocessor that handles 20 Newsgroups specific artefacts.
    """

    def __init__(self, chunk_size: int = 100, overlap: int = 25,
                 min_chunk_words: int = 20, min_doc_chars: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_words = min_chunk_words
        self.min_doc_chars = min_doc_chars

        # Precompile regular expressions for speed
        self.re_email = re.compile(r'\S+@\S+\.\S+')
        self.re_url = re.compile(r'https?://\S+')
        self.re_non_ascii = re.compile(r'[^\x00-\x7F]+')
        self.re_whitespace = re.compile(r'\s+')

    def parse_doc(self, raw_text: str) -> Tuple[str, str]:
        """
        Strips header block and keeps only the Subject line.
        """
        lines = raw_text.splitlines()
        subject = "No Subject"
        body_start_idx = 0

        for i, line in enumerate(lines):
            if line.startswith("Subject:"):
                subject = line[8:].strip()
            # First completely blank line normally separates header and body
            if not line.strip():
                body_start_idx = i + 1
                break

        if body_start_idx == 0:
            # Fallback if no blank line is found
            body_start_idx = len(lines)

        body = "\n".join(lines[body_start_idx:])
        return subject, body

    def clean(self, text: str) -> str:
        """
        Cleans quoted lines, signatures, emails, URLs, non-ascii chars,
        and normalises the text string.
        """
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            # Remove the signature block (anything after '--' on its own line)
            if line.strip() == '--':
                break

            stripped = line.strip()
            # Remove standard quoted text symbols
            if stripped.startswith('>') or stripped.startswith('|'):
                continue

            cleaned_lines.append(line)

        joined_text = " ".join(cleaned_lines)

        # Regex application for noisy artefacts
        joined_text = self.re_email.sub('', joined_text)
        joined_text = self.re_url.sub('', joined_text)
        joined_text = self.re_non_ascii.sub('', joined_text)
        joined_text = self.re_whitespace.sub(' ', joined_text)

        return joined_text.lower().strip()

    def chunk(self, text: str) -> List[str]:
        """
        Implements sliding window chunking by words constraint.
        """
        words = text.split()
        chunks = []

        if not words:
            return chunks

        step = self.chunk_size - self.overlap
        if step <= 0:
            step = 1  # Fallback safety

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) >= self.min_chunk_words:
                chunks.append(" ".join(chunk_words))

        return chunks

    def process_file(self, path: Path) -> Optional[Dict[str, any]]:
        """
        Runs the full data pipeline on a single raw text file.
        Returns None if the document failed the length gating logic.
        """
        try:
            # latin1 encoding handles older datasets better
            with open(path, 'r', encoding='latin1') as f:
                raw_text = f.read()
        except Exception:
            return None

        subject, body = self.parse_doc(raw_text)
        cleaned_body = self.clean(body)

        # Gate check for uselessly short documents
        if len(cleaned_body) < self.min_doc_chars:
            return None

        chunks = self.chunk(cleaned_body)
        if not chunks:
            return None

        return {
            "subject": subject,
            "text": cleaned_body,
            "chunks": chunks
        }

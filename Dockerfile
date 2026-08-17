FROM python:3.12-slim

WORKDIR /app

# OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE work around a duplicate-OpenMP-
# runtime crash between torch and numpy/scipy's own OpenMP-linked BLAS when
# both are loaded in the same process (hit during Phase 2 development on
# Intel macOS; harmless to set unconditionally, including on Linux).
ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

ENV PIP_DEFAULT_TIMEOUT=180

COPY requirements.txt .
# torch is by far the largest wheel here and the most likely to hit a slow-
# network read timeout; installing it in its own layer means a flaky retry
# doesn't have to re-download everything else that already succeeded.
RUN grep -v '^torch' requirements.txt > /tmp/requirements-rest.txt \
    && pip install --no-cache-dir -r /tmp/requirements-rest.txt
RUN pip install --no-cache-dir torch>=2.2

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

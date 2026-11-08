FROM python:3.10.16-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY experiments/obp/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir --no-deps obp==0.5.7

COPY experiments/obp/patch_obp.py /tmp/patch_obp.py
RUN python /tmp/patch_obp.py

CMD ["python", "-m", "experiments.obp.run"]

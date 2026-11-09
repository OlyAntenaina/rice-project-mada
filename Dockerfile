# Mampiasa Python 3.11
FROM python:3.11-slim

# Mamorona ny dossier ao anaty container
WORKDIR /app

# Mandika ny requirements
COPY requirements.txt .

# Mametraka ny bibliothèques
RUN pip install --no-cache-dir -r requirements.txt

# Mandika ny code rehetra
COPY . .

# Manokatra ny port 8000
EXPOSE 8000

# Mampandeha ny API
CMD ["uvicorn", "services.api:app", "--host", "0.0.0.0", "--port", "8000"]
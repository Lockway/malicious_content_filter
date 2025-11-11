# Malicious Content Filtering

## Local Test
```
pip install -r requirements.txt
uvicorn main:app --reload
```

## Request
```
{
    'text': "안녕하세요"
}
```

## Response
```
{
    'hate': 0.01,
    'spam': 0.01,
}
```


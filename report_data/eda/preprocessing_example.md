# Exemple de Prétraitement d'un CV

### 1. Texte Brut (300 premiers caractères)
```text
SUMMARY:
Dedicated and result-oriented Staff Electrical Engineering Professional with over 8 years of comprehensive experience in electrical systems design, circuit analysis, power distribution, plc programming, and matlab.
Proven track record of driving impactful projects, delivering high quality r
```

### 2. Nettoyage et Minuscules (Sans ponctuation)
```text
summary 
dedicated and result oriented staff electrical engineering professional with over 8 years of comprehensive experience in electrical systems design  circuit analysis  power distribution  plc programming  and matlab 
proven track record of driving impactful projects  delivering high quality r
```

### 3. Suppression des Stop Words
```text
summary dedicated result oriented staff electrical engineering professional 8 years comprehensive experience electrical systems design circuit analysis power distribution plc programming matlab proven track record driving impactful projects delivering high quality r
```

### 4. Tokenisation
```json
["summary", "dedicated", "result", "oriented", "staff", "electrical", "engineering", "professional", "8", "years", "comprehensive", "experience", "electrical", "systems", "design", "circuit", "analysis", "power", "distribution", "plc", "programming", "matlab", "proven", "track", "record", "driving", "impactful", "projects", "delivering", "high", "quality", "r"]
```

### 5. Lemmatisation
```json
["summary", "dedicated", "result", "oriented", "staff", "electrical", "engineering", "professional", "8", "year", "comprehensive", "experience", "electrical", "system", "design", "circuit", "analysis", "power", "distribution", "plc", "programming", "matlab", "proven", "track", "record", "driving", "impactful", "project", "delivering", "high", "quality", "r"]
```


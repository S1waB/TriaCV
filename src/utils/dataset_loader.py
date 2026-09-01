"""Dataset loader, downloader, and synthetic corpus generator for Kaggle Resume Dataset (~25 categories)."""
import os
import random
from typing import Dict, List, Tuple
import pandas as pd
import requests

# 25 Official Categories of the Kaggle Resume Dataset
CATEGORIES_INFO: Dict[str, Dict[str, any]] = {
    "Data Science": {
        "description": "Machine learning, statistical analysis, deep learning, NLP, and predictive modeling.",
        "skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NLP", "Computer Vision", "SQL", "Tableau", "Statistics", "Big Data", "Feature Engineering", "Data Visualization"]
    },
    "HR": {
        "description": "Human resources management, talent acquisition, employee relations, onboarding, and payroll.",
        "skills": ["Recruitment", "Talent Acquisition", "Employee Engagement", "Performance Management", "Payroll", "HR Policies", "Onboarding", "Labor Laws", "HRIS", "Compensation & Benefits", "Conflict Resolution", "Interviews"]
    },
    "Advocate": {
        "description": "Legal counsel, courtroom litigation, contract drafting, corporate compliance, and arbitration.",
        "skills": ["Litigation", "Legal Research", "Contract Drafting", "Corporate Law", "Arbitration", "Court Hearings", "Client Counseling", "Intellectual Property", "Legal Due Diligence", "Compliance", "Statutory Interpretation"]
    },
    "Arts": {
        "description": "Visual arts, graphic design, illustration, art direction, exhibitions, and creative styling.",
        "skills": ["Graphic Design", "Illustration", "Adobe Photoshop", "Illustrator", "Art Direction", "Exhibitions", "Typography", "Visual Storytelling", "Painting", "Digital Art", "Color Theory", "Creative Direction"]
    },
    "Web Designing": {
        "description": "Front-end web development, UI/UX design, responsive wireframes, HTML/CSS, and Javascript.",
        "skills": ["HTML5", "CSS3", "JavaScript", "Responsive Design", "Figma", "UI/UX Design", "Bootstrap", "SASS", "Wireframing", "Webflow", "Adobe XD", "Cross-browser Compatibility", "DOM Manipulation"]
    },
    "Mechanical Engineer": {
        "description": "Mechanical design, CAD modeling, thermodynamics, manufacturing processes, and structural analysis.",
        "skills": ["SolidWorks", "AutoCAD", "CATIA", "Thermodynamics", "ANSYS", "Finite Element Analysis", "Manufacturing Processes", "Pneumatics", "Hydraulics", "GD&T", "HVAC", "CNC Machining"]
    },
    "Sales": {
        "description": "Business development, account management, client prospecting, B2B sales, and revenue growth.",
        "skills": ["B2B Sales", "Lead Generation", "Client Relationship", "Negotiation", "Sales Pipeline", "CRM", "Salesforce", "Revenue Growth", "Cold Calling", "Account Management", "Market Expansion", "Closing Deals"]
    },
    "Health and fitness": {
        "description": "Physical training, nutritional planning, wellness coaching, kinesiology, and rehab exercise.",
        "skills": ["Personal Training", "Nutrition Planning", "Fitness Assessment", "Strength Conditioning", "Kinesiology", "Rehabilitation", "Weight Management", "Aerobics", "Wellness Coaching", "CPR Certified", "Cardio Training"]
    },
    "Civil Engineer": {
        "description": "Structural engineering, construction management, site surveying, geotechnical analysis, and AutoCAD.",
        "skills": ["AutoCAD", "Civil 3D", "Structural Analysis", "STAAD Pro", "Construction Management", "Site Surveying", "Concrete Technology", "Geotechnical Engineering", "Cost Estimation", "BIM", "Project Scheduling"]
    },
    "Java Developer": {
        "description": "Backend software engineering with Java, Spring Boot, Microservices, Hibernate, and REST APIs.",
        "skills": ["Java", "Spring Boot", "Spring MVC", "Hibernate", "Microservices", "RESTful APIs", "Maven", "JPA", "JUnit", "SQL", "Kafka", "Multithreading", "Docker", "Design Patterns"]
    },
    "Business Analyst": {
        "description": "Business requirements gathering, data modeling, process re-engineering, Agile/Scrum, and BI.",
        "skills": ["Requirements Gathering", "BRD", "FRD", "Agile / Scrum", "Data Analysis", "SQL", "Tableau", "Power BI", "User Stories", "Stakeholder Management", "Process Mapping", "GAP Analysis"]
    },
    "SAP Developer": {
        "description": "SAP ERP consulting, ABAP development, Fiori/UI5, SAP HANA, and enterprise modules.",
        "skills": ["SAP ABAP", "SAP HANA", "SAP Fiori", "SAP ERP", "IDoc", "BAPI", "OData Services", "SAP SD", "SAP MM", "SAP FI/CO", "ALV Reports", "Enhancements & BADIs"]
    },
    "Automation Testing": {
        "description": "Automated software QA, Selenium, test frameworks, CI/CD test pipelines, and bug tracking.",
        "skills": ["Selenium WebDriver", "TestNG", "Cucumber BDD", "Java", "Python", "API Testing", "Postman", "Jenkins", "Jira", "Regression Testing", "Appium", "XPath", "Page Object Model"]
    },
    "Electrical Engineering": {
        "description": "Electrical systems design, circuit analysis, power distribution, PLC programming, and MATLAB.",
        "skills": ["Circuit Design", "Power Systems", "PLC Programming", "SCADA", "MATLAB / Simulink", "AutoCAD Electrical", "Electrical Schematics", "Transformer", "Switchgear", "Instrumentation", "Microcontrollers"]
    },
    "Operations Manager": {
        "description": "Operations management, supply chain optimization, process excellence, Lean Six Sigma, and logistics.",
        "skills": ["Operations Management", "Supply Chain", "Process Improvement", "Lean Six Sigma", "Logistics", "Inventory Control", "Budgeting", "Vendor Management", "KPI Monitoring", "Quality Assurance", "Cost Reduction"]
    },
    "Python Developer": {
        "description": "Backend and web application development using Python, Django, Flask, FastAPI, and PostgreSQL.",
        "skills": ["Python", "Django", "Flask", "FastAPI", "PostgreSQL", "REST APIs", "Celery", "Redis", "SQLAlchemy", "Asyncio", "Docker", "Git", "Pytest", "Object-Oriented Programming"]
    },
    "DevOps Engineer": {
        "description": "CI/CD pipelines, container orchestration, cloud infrastructure, Kubernetes, Terraform, and Docker.",
        "skills": ["Docker", "Kubernetes", "CI/CD Pipelines", "Jenkins", "Terraform", "Ansible", "AWS", "Azure", "Linux Shell Scripting", "Prometheus", "Grafana", "GitOps", "Infrastructure as Code"]
    },
    "Network Security Engineer": {
        "description": "Cybersecurity, network protocols, firewalls, intrusion detection, vulnerability assessment, and SIEM.",
        "skills": ["Network Security", "Firewalls", "Cisco", "Routing & Switching", "Wireshark", "Vulnerability Assessment", "Penetration Testing", "SIEM", "IDS/IPS", "VPN", "CompTIA Security+", "Incident Response"]
    },
    "PMO": {
        "description": "Project Management Office, portfolio tracking, risk management, governance, PMP, and budgeting.",
        "skills": ["Project Governance", "Portfolio Management", "Risk Management", "MS Project", "Stakeholder Communication", "Budget Tracking", "PMP", "Resource Allocation", "Reporting & Dashboards", "Milestone Tracking"]
    },
    "Database": {
        "description": "Database administration, SQL optimization, schema design, backup/recovery, Oracle, and MySQL.",
        "skills": ["SQL", "Database Administration", "MySQL", "Oracle DB", "PostgreSQL", "Query Optimization", "Database Indexing", "Backup & Recovery", "Stored Procedures", "Data Migration", "ETL", "NoSQL / MongoDB"]
    },
    "Hadoop": {
        "description": "Big data distributed architectures, Hadoop ecosystem, MapReduce, HDFS, Hive, Spark, and Pig.",
        "skills": ["Hadoop", "HDFS", "MapReduce", "Apache Spark", "Hive", "Pig", "HBase", "Sqoop", "Flume", "Big Data", "Kafka", "Scala", "YARN", "Data Lake"]
    },
    "ETL Developer": {
        "description": "Extract, Transform, Load data pipelines, Informatica, Talend, data warehousing, and SSIS.",
        "skills": ["ETL", "Informatica PowerCenter", "Talend", "SSIS", "Data Warehousing", "SQL", "Star Schema", "Data Modeling", "PL/SQL", "Data Cleansing", "Pipeline Orchestration", "Dimension Modeling"]
    },
    "DotNet Developer": {
        "description": ".NET framework development, C#, ASP.NET Core, Entity Framework, Web API, and SQL Server.",
        "skills": ["C#", ".NET Core", "ASP.NET MVC", "Entity Framework", "Web API", "SQL Server", "LINQ", "WPF", "Microservices", "T-SQL", "Azure", "Dependency Injection", "Unit Testing"]
    },
    "Blockchain": {
        "description": "Decentralized applications, Ethereum smart contracts, Solidity, Web3.js, Hyperledger, and cryptography.",
        "skills": ["Blockchain", "Solidity", "Ethereum", "Smart Contracts", "Web3.js", "Hyperledger Fabric", "Cryptography", "Decentralized Apps (dApps)", "Truffle", "Hardhat", "Consensus Algorithms", "Tokenomics"]
    },
    "Journalist": {
        "description": "Reporting, news writing, storytelling, media.",
        "skills": ["Reporting", "Journalism", "News Writing", "Media", "Press"]
    },
    "Chief Editor": {
        "description": "Editorial strategy, content management, publishing.",
        "skills": ["Publishing", "Content Management", "Editorial", "Copy Editing", "SEO"]
    },
    "Graphic Designer": {
        "description": "Visual arts, UI/UX, branding.",
        "skills": ["Adobe Photoshop", "Illustrator", "Figma", "UI/UX", "Branding"]
    },
    "Doctor": {
        "description": "Medicine, patient care, healthcare.",
        "skills": ["Medicine", "Patient Care", "Healthcare", "Surgery", "Clinical"]
    },
    "Content Writer": {
        "description": "Copywriting, blogging, SEO.",
        "skills": ["Copywriting", "SEO", "Blogging", "Writing", "Social Media"]
    },
    "Content Creator": {
        "description": "Video editing, social media marketing.",
        "skills": ["Video Editing", "Social Media", "TikTok", "Instagram", "Marketing"]
    },
    "YouTuber": {
        "description": "Vlogging, YouTube Analytics, Monetization.",
        "skills": ["YouTube", "Video Production", "Vlogging", "Monetization", "OBS"]
    },
    "Gamer": {
        "description": "eSports, Twitch streaming, competitive gaming.",
        "skills": ["eSports", "Twitch", "OBS", "Gaming", "Discord"]
    },
    "Testing": {
        "description": "Quality assurance, manual testing, test case design, defect tracking, black-box testing, and SDLC.",
        "skills": ["Manual Testing", "Test Case Design", "Defect Life Cycle", "Jira", "Regression Testing", "Functional Testing", "System Testing", "UAT", "STLC", "Bugzilla", "Test Execution", "API Testing"]
    }
}

# Public dataset mirrors
DATASET_URLS = [
    "https://raw.githubusercontent.com/manikanta-sandepudi/Resume-Screening-and-Classification/master/UpdatedResumeDataSet.csv",
    "https://raw.githubusercontent.com/dhruv-anand-ainur/Resume-Classification-NLP/master/UpdatedResumeDataSet.csv",
    "https://raw.githubusercontent.com/saurabh-k-singh/Resume-Screening/main/UpdatedResumeDataSet.csv"
]


def download_kaggle_dataset(target_path: str) -> bool:
    """Attempt downloading UpdatedResumeDataSet.csv from public mirrors."""
    for url in DATASET_URLS:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and "Category" in resp.text and "Resume" in resp.text:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                return True
        except Exception:
            continue
    return False


def generate_synthetic_dataset(num_samples_per_category: int = 50) -> pd.DataFrame:
    """Generate a realistic synthetic resume dataset across all 25 categories."""
    random.seed(42)
    records = []

    institutions = ["University of Technology", "State Institute of Science", "National Polytechnic", "City University", "College of Engineering", "Institute of Management"]
    companies = ["Global Tech Solutions", "Apex Innovations", "Nexus Consulting", "InnoWave Systems", "Pinnacle Enterprise", "Alpha Dynamics", "Vanguard Industries"]
    job_titles_prefix = ["Senior", "Lead", "Associate", "Staff", "Principal", "Specialist", "Consultant"]

    action_verbs = [
        "Architected and implemented", "Spearheaded the development of", "Managed and delivered",
        "Optimized and maintained", "Collaborated with cross-functional teams to design",
        "Streamlined workflows and improved", "Conducted in-depth analysis and deployed",
        "Engineered scalable solutions utilizing", "Supervised operational tasks and implemented"
    ]

    for category, meta in CATEGORIES_INFO.items():
        skills = meta["skills"]
        description = meta["description"]

        for idx in range(num_samples_per_category):
            chosen_skills = random.sample(skills, min(len(skills), random.randint(7, 12)))
            extra_skills = random.sample(skills, random.randint(3, 5))
            years_exp = random.randint(2, 12)
            title = f"{random.choice(job_titles_prefix)} {category} Professional"
            company1 = random.choice(companies)
            company2 = random.choice(companies)
            school = random.choice(institutions)

            resume_lines = [
                f"SUMMARY:",
                f"Dedicated and result-oriented {title} with over {years_exp} years of comprehensive experience in {description.lower()}",
                f"Proven track record of driving impactful projects, delivering high quality results, and mastering {', '.join(chosen_skills[:4])}.",
                "",
                f"TECHNICAL SKILLS & COMPETENCIES:",
                f"Core Skills: {', '.join(chosen_skills)}",
                f"Methodologies & Tools: {', '.join(extra_skills)}, Agile, Git, Jira, CI/CD, Problem Solving, Analytical Thinking.",
                "",
                f"WORK EXPERIENCE:",
                f"1. {title} | {company1} (2020 - Present)",
                f"- {random.choice(action_verbs)} enterprise systems using {chosen_skills[0]} and {chosen_skills[1]}.",
                f"- Increased operational efficiency by {random.randint(15, 45)}% through hands-on implementation of {chosen_skills[2]} and {chosen_skills[3]}.",
                f"- Led a team of {random.randint(3, 10)} engineers to successfully deliver mission-critical milestones within strict deadlines.",
                f"- Authored technical documentation, maintained coding standards, and reviewed code for reliability and security.",
                "",
                f"2. {category} Specialist | {company2} (2016 - 2020)",
                f"- {random.choice(action_verbs)} solutions based on {chosen_skills[-1]} and {chosen_skills[-2]}.",
                f"- Handled day-to-day requirements, troubleshooting, and continuous improvement for enterprise stakeholders.",
                f"- Received Employee of the Quarter award for exceptional delivery and innovative contributions.",
                "",
                f"EDUCATION & CERTIFICATIONS:",
                f"- Bachelor of Science / Engineering in Computer Science / Relevant Field, {school} (Graduated with Honors).",
                f"- Certified Professional in {chosen_skills[0]} & {category} Best Practices.",
                "",
                f"KEY PROJECTS:",
                f"- Enterprise {category} Transformation: Successfully migrated legacy workflows into modern high-throughput pipelines.",
                f"- Automated Workflow Initiative: Integrated {chosen_skills[0]} and {chosen_skills[1]} to streamline business operations."
            ]

            full_resume = "\n".join(resume_lines)
            records.append({"Category": category, "Resume": full_resume})

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def _load_or_create_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load raw dataset from disk, download from mirror, or generate high-quality fallback corpus."""
    os.makedirs(raw_dir, exist_ok=True)
    raw_file = os.path.join(raw_dir, "UpdatedResumeDataSet.csv")

    if os.path.exists(raw_file):
        try:
            df = pd.read_csv(raw_file)
            if "Category" in df.columns and "Resume" in df.columns and len(df) > 100:
                print(f"[DatasetLoader] Loaded existing dataset from {raw_file} ({len(df)} samples, {df['Category'].nunique()} categories).")
                return df
        except Exception as e:
            print(f"[DatasetLoader] Could not read existing file: {e}")

    # Try downloading
    print("[DatasetLoader] Attempting to download Kaggle Resume Dataset from public mirrors...")
    downloaded = download_kaggle_dataset(raw_file)
    if downloaded:
        try:
            df = pd.read_csv(raw_file)
            if "Category" in df.columns and "Resume" in df.columns and len(df) > 100:
                print(f"[DatasetLoader] Successfully downloaded dataset ({len(df)} samples, {df['Category'].nunique()} categories).")
                return df
        except Exception as e:
            print(f"[DatasetLoader] Download succeeded but parsing failed: {e}")

    # Fallback to high-quality synthetic generation
    print("[DatasetLoader] Generating synthetic 25-category Resume Dataset (~1250 samples)...")
    df = generate_synthetic_dataset(num_samples_per_category=50)
    df.to_csv(raw_file, index=False, encoding="utf-8")
    print(f"[DatasetLoader] Synthetic dataset saved to {raw_file} ({len(df)} samples across {df['Category'].nunique()} categories).")
    return df


if __name__ == "__main__":
    df_data = load_or_create_dataset()
    print("Dataset sample:")
    print(df_data.head())
    print("Category value counts:")
    print(df_data["Category"].value_counts())


def load_or_create_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
    df = _load_or_create_dataset(raw_dir)
    existing_cats = set(df["Category"].unique())
    missing = [c for c in CATEGORIES_INFO if c not in existing_cats]
    if missing:
        import random
        print(f"[DatasetLoader] Injecting synthetic data for {len(missing)} missing categories...")
        new_rows = []
        for cat in missing:
            meta = CATEGORIES_INFO[cat]
            skills = meta["skills"]
            for _ in range(100):
                chosen_skills = random.sample(skills, k=min(6, len(skills)))
                resume = f"Summary: Experienced {cat} professional. Skills: {', '.join(chosen_skills)}. "
                resume += f"Experience: Senior {cat} at TechCorp. Optimized and maintained {chosen_skills[0]} solutions."
                new_rows.append({"Category": cat, "Resume": resume})
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df
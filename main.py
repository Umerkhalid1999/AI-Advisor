import streamlit as st
import openai
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
import json
import altair as alt

# Add this right after your imports in main.py
# --------------------------
# 🛠️ Configuration & Setup
# --------------------------
st.set_page_config(
    page_title="Smart College Advisor Pro",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)


# --------------------------
# 🔐 Authentication (Simulated)
# --------------------------
def check_auth():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    return st.session_state.authenticated


def login():
    with st.sidebar:
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # Simple demo authentication
            if email and password:  # In real app, verify against database
                st.session_state.authenticated = True
                st.rerun()


# --------------------------
# 🗃️ Database Setup (SQLite)
# --------------------------
def init_db():
    conn = sqlite3.connect('student_data.db')
    c = conn.cursor()

    # Student responses table
    c.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            student_id TEXT,
            responses TEXT,
            recommendation TEXT,
            ai_insights TEXT
        )
    ''')

    # Admin/counselor table
    c.execute('''
        CREATE TABLE IF NOT EXISTS counselors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            specialization TEXT,
            availability TEXT
        )
    ''')

    # Case studies table
    c.execute('''
        CREATE TABLE IF NOT EXISTS case_studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            description TEXT,
            outcome TEXT,
            tags TEXT
        )
    ''')

    conn.commit()
    conn.close()


init_db()


# --------------------------
# 🧠 AI Configuration (GPT-3.5)
# --------------------------
import streamlit as st
from openai import OpenAI  # Updated import for new API version
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
import json
import altair as alt

# Initialize OpenAI client
client = OpenAI(st.secrets["OPENAI_API_KEY"])  # Will use environment variable OPENAI_API_KEY

# --------------------------
# 🧠 Updated AI Configuration (GPT-3.5)
# --------------------------
def get_ai_response(prompt, context=""):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""
                You are an advanced college advisor AI. Provide:
                - Specific major recommendations based on student profile
                - Career path insights with salary ranges
                - Skill development roadmap
                - Comparative analysis with traditional methods
                Context: {context}
                """},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=250
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --------------------------
# 🤖 AI Supervisor Chatbot
# --------------------------
# Replace the entire ai_supervisor_chat() function with this:

def ai_supervisor_chat():
    # Page title
    st.markdown('<h1 class="main-header">💬 AI Supervisor Chat</h1>', unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask the AI Supervisor..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            # Build context from assessment results if available
            context = ""
            if 'student_id' in st.session_state:
                context += f"Student ID: {st.session_state.student_id}. "
            if 'assessment_done' in st.session_state and st.session_state.assessment_done:
                scores = calculate_scores(st.session_state.responses)
                top_category = max(scores, key=scores.get)
                context += f"Assessment results: Top category is {top_category} with score {scores[top_category]:.1f}. "
                context += f"Learning style: {st.session_state.responses.get('learning_style', 'Not specified')}. "
                context += f"Career values: {st.session_state.responses.get('career_values', [])}."

            # Stream response from OpenAI
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": f"You are an AI college advisor. Provide helpful, specific advice about majors, careers, and academic planning. Context: {context}"},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ],
                stream=True,
            )

            # Process the streaming response
            for chunk in stream:
                if chunk_content := chunk.choices[0].delta.content:
                    full_response += chunk_content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Add clear chat button at the bottom
    if st.session_state.messages and st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
# --------------------------
# 📊 Adaptive Questionnaire Engine
# --------------------------
QUESTIONS = [
    {
        "id": "academic_performance",
        "question": "Rate your academic performance in core subjects (Math, Science, Language):",
        "type": "likert",
        "options": ["Very weak", "Weak", "Average", "Strong", "Very strong"],
        "weights": {"STEM": [0.1, 0.3, 0.5, 0.8, 1.0], "Humanities": [0.2, 0.4, 0.6, 0.7, 0.9]}
    },
    {
        "id": "learning_style",
        "question": "How do you prefer to learn?",
        "type": "mcq",
        "options": [
            "Through experiments/hands-on work (Kinesthetic)",
            "By reading/writing (Verbal)",
            "Through visual aids (Visual)",
            "Via discussions (Auditory)"
        ],
        "branching": {
            "Kinesthetic": "stem_followup",
            "Visual": "arts_followup"
        }
    },
    {
        "id": "stem_followup",
        "question": "Do you enjoy solving complex technical problems?",
        "type": "likert",
        "condition": ("learning_style", "Kinesthetic"),
        "weights": {"STEM": [0.3, 0.5, 0.7, 0.9, 1.0]}
    },
    {
        "id": "career_values",
        "question": "What matters most in your future career?",
        "type": "rank",
        "options": ["High salary", "Work-life balance", "Creativity", "Social impact"],
        "weights": {
            "High salary": {"Business": 0.8, "STEM": 0.7},
            "Creativity": {"Arts": 1.0, "Humanities": 0.6}
        }
    },
     {
        "id": "humanities_interest",
        "question": "How interested are you in humanities subjects?",
        "type": "likert",
        "options": ["Not at all", "Slightly", "Moderately", "Very", "Extremely"],
        "weights": {"Humanities": [0.1, 0.3, 0.5, 0.8, 1.0]}
    }
]

MAJOR_CATEGORIES = {
    "STEM": {
        "majors": ["Computer Science", "Engineering", "Data Science"],
        "skills": ["Programming", "Analytical Thinking", "Mathematics"],
        "careers": {
            "Software Engineer": {"salary": "$70k-$150k", "growth": "22% (2020-2030)"},
            "Data Scientist": {"salary": "$85k-$170k", "growth": "31% (2020-2030)"}
        }
    },
    "Business": {
        "majors": ["Finance", "Marketing", "Entrepreneurship"],
        "skills": ["Leadership", "Communication", "Financial Literacy"],
        "careers": {
            "Financial Analyst": {"salary": "$60k-$120k", "growth": "6% (2020-2030)"},
            "Marketing Manager": {"salary": "$65k-$135k", "growth": "10% (2020-2030)"}
        }
    },
    "Humanities": {  # Added this missing category
        "majors": ["Psychology", "History", "Philosophy"],
        "skills": ["Writing", "Critical Thinking", "Research"],
        "careers": {
            "Psychologist": {"salary": "$50k-$120k", "growth": "8% (2020-2030)"},
            "Historian": {"salary": "$45k-$90k", "growth": "5% (2020-2030)"}
        }
    },
    "Arts": {  # Added this missing category
        "majors": ["Graphic Design", "Music", "Film"],
        "skills": ["Creativity", "Visual Communication", "Performance"],
        "careers": {
            "Graphic Designer": {"salary": "$40k-$85k", "growth": "3% (2020-2030)"},
            "Musician": {"salary": "$30k-$100k", "growth": "4% (2020-2030)"}
        }
    }
}


# --------------------------
# 🔍 Scoring & Recommendation Engine
# --------------------------
def calculate_scores(responses):
    # Initialize all categories with 0 score
    scores = {category: 0 for category in MAJOR_CATEGORIES.keys()}

    for q in QUESTIONS:
        if q["id"] in responses:
            if q["type"] == "likert":
                for category, weights in q.get("weights", {}).items():
                    # Only process categories that exist in MAJOR_CATEGORIES
                    if category in MAJOR_CATEGORIES:
                        try:
                            response_index = responses[q["id"]]
                            if isinstance(weights, list) and len(weights) > response_index:
                                scores[category] += weights[response_index] * 20
                        except (IndexError, KeyError) as e:
                            print(f"Warning: Error processing {q['id']} for {category}: {e}")

            elif q["type"] == "rank":
                for option in responses[q["id"]]:
                    for category, weight in q.get("weights", {}).get(option, {}).items():
                        if category in MAJOR_CATEGORIES:
                            try:
                                scores[category] += weight * (4 - responses[q["id"]].index(option))
                            except (ValueError, KeyError) as e:
                                print(f"Warning: Error processing ranking for {option}: {e}")

    # Normalize to 0-100 scale
    max_score = max(scores.values()) if max(scores.values()) > 0 else 1
    return {k: (v / max_score * 100) for k, v in scores.items()}


# --------------------------
# 📊 Visualization Tools
# --------------------------
def plot_radar_chart(scores):
    categories = list(scores.keys())
    values = list(scores.values())

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='#4B8BBE', linewidth=2)
    ax.fill(angles, values, color='#4B8BBE', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_title("Career Pathway Affinity", pad=20)
    st.pyplot(fig)


def plot_career_salary(category):
    careers = MAJOR_CATEGORIES[category]["careers"]
    data = pd.DataFrame([
        {"Career": career, "Salary Range": info["salary"], "Growth": info["growth"]}
        for career, info in careers.items()
    ])

    chart = alt.Chart(data).mark_bar().encode(
        x='Career:N',
        y=alt.Y('Salary Range:Q', scale=alt.Scale(domain=[0, 200000])),
        color='Growth:N',
        tooltip=['Career', 'Salary Range', 'Growth']
    ).properties(width=600)
    st.altair_chart(chart)


# --------------------------
# 👨‍🏫 Counselor Dashboard
# --------------------------
def counselor_dashboard():
    st.header("👩‍🏫 Counselor Dashboard")

    tab1, tab2, tab3 = st.tabs(["Student Insights", "Case Studies", "Availability"])

    with tab1:
        conn = sqlite3.connect('student_data.db')
        df = pd.read_sql("SELECT * FROM responses", conn)
        conn.close()

        if not df.empty:
            st.subheader("Student Analytics")
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Convert responses from string to dict
            df['responses_dict'] = df['responses'].apply(lambda x: eval(x) if x else {})

            # Show summary stats
            st.metric("Total Students", len(df))
            st.metric("Most Common Recommendation", df['recommendation'].mode()[0])

            # Show response trends
            st.subheader("Response Trends")
            time_chart = alt.Chart(df).mark_line().encode(
                x='timestamp:T',
                y='count():Q'
            )
            st.altair_chart(time_chart, use_container_width=True)
        else:
            st.warning("No student data available yet")


# --------------------------
# 🖥️ Main Application
# --------------------------
def student_application():
    # Initialize session state
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'current_q' not in st.session_state:
        st.session_state.current_q = 0
        # Add these new session state variables
        if 'recommendations_generated' not in st.session_state:
            st.session_state.recommendations_generated = False
        if 'ai_recommendation' not in st.session_state:
            st.session_state.ai_recommendation = None
        if 'scores' not in st.session_state:
            st.session_state.scores = None
        if 'top_category' not in st.session_state:
            st.session_state.top_category = None


    # Student ID handling
    if 'student_id' not in st.session_state:
        st.title("🎓 Smart College Advisor Pro")
        st.caption("AI-Powered Career Pathway Analysis")

        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                st.image("student_icon.png",
                         width=100, caption="Student Icon")
            except:
                st.markdown("""
                <style>.big-emoji {font-size: 50px;}</style>
                <div class="big-emoji">🎓</div>
                """, unsafe_allow_html=True)

        with col2:
            student_id_input = st.text_input("Enter your student ID:")
            if st.button("Continue") and student_id_input:
                st.session_state.student_id = student_id_input
                st.rerun()
        return

    # Questionnaire Flow
    if not st.session_state.get("assessment_done"):
        with st.expander("📝 Personalized Assessment", expanded=True):
            if st.session_state.current_q < len(QUESTIONS):
                q = QUESTIONS[st.session_state.current_q]

                # Handle branching logic
                if "condition" in q:
                    key, value = q["condition"]
                    if st.session_state.responses.get(key) != value:
                        st.session_state.current_q += 1
                        st.rerun()

                st.progress(st.session_state.current_q / len(QUESTIONS))
                st.subheader(q["question"])

                if q["type"] == "likert":
                    response = st.radio(
                        "Select:",
                        q["options"],
                        key=q["id"],
                        horizontal=True
                    )
                    st.session_state.responses[q["id"]] = q["options"].index(response)

                elif q["type"] == "rank":
                    st.write("Rank from most (1) to least (4) important:")
                    ranked = []
                    for i, option in enumerate(q["options"]):
                        ranked.append(st.selectbox(
                            option,
                            [1, 2, 3, 4],
                            key=f"rank_{q['id']}_{i}"
                        ))
                    st.session_state.responses[q["id"]] = [
                        x[1] for x in sorted(zip(ranked, q["options"]))
                    ]

                # Navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.session_state.current_q > 0 and st.button("← Back"):
                        st.session_state.current_q -= 1
                        st.rerun()
                with col2:
                    if st.button("Next →"):
                        st.session_state.current_q += 1
                        st.rerun()
            else:
                st.success("✅ Assessment Complete!")
                st.session_state.assessment_done = True
                st.rerun()

        # Results Dashboard
    else:
        try:
            st.header("Your Career Pathway Analysis")

            # Calculate scores with error handling - only if not already calculated
            if st.session_state.scores is None:
                st.session_state.scores = calculate_scores(st.session_state.responses)
                st.session_state.top_category = max(st.session_state.scores, key=st.session_state.scores.get)

            # Validate top category exists
            if st.session_state.top_category not in MAJOR_CATEGORIES:
                st.error("Unexpected category result. Please try the assessment again.")
                st.session_state.assessment_done = False
                st.rerun()

            # Visualization
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Affinity Scores")
                plot_radar_chart(st.session_state.scores)
            with col2:
                st.subheader("Recommended Majors")
                for major in MAJOR_CATEGORIES[st.session_state.top_category]["majors"]:
                    with st.expander(major):
                        st.write(
                            f"**Key Skills:** {', '.join(MAJOR_CATEGORIES[st.session_state.top_category]['skills'])}")
                        st.write("**Career Opportunities:**")
                        for career, info in MAJOR_CATEGORIES[st.session_state.top_category]["careers"].items():
                            st.write(f"- {career} (Salary: {info['salary']}, Growth: {info['growth']})")

            # AI-Powered Insights - only generate if not already generated
            if not st.session_state.recommendations_generated:
                with st.spinner("🔍 Generating personalized insights..."):
                    context = {
                        "top_category": st.session_state.top_category,
                        "score": st.session_state.scores[st.session_state.top_category],
                        "learning_style": st.session_state.responses.get('learning_style', 'Not specified'),
                        "career_values": st.session_state.responses.get('career_values', [])
                    }

                    st.session_state.ai_recommendation = get_ai_response(
                        f"Provide detailed analysis for a {context['learning_style']} learner interested in {st.session_state.top_category}. "
                        "Include: 1) Recommended courses, 2) Career paths with salary ranges, "
                        "3) Skill development plan, 4) Job market outlook",
                        str(context)
                    )
                    st.session_state.recommendations_generated = True

            # Display the recommendations (whether just generated or from session state)
            st.markdown("### 🎯 AI-Powered Recommendations")
            st.markdown(st.session_state.ai_recommendation)

            # Database Storage - only save if we just generated recommendations
            if st.session_state.recommendations_generated:
                try:
                    conn = sqlite3.connect('student_data.db')
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO responses (timestamp, student_id, responses, recommendation, ai_insights) VALUES (?, ?, ?, ?, ?)",
                        (datetime.now(), st.session_state.student_id,
                         json.dumps(st.session_state.responses),
                         st.session_state.top_category, st.session_state.ai_recommendation)
                    )
                    conn.commit()
                except sqlite3.Error as e:
                    st.warning(f"Could not save results to database: {str(e)}")
                finally:
                    if conn:
                        conn.close()

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.session_state.assessment_done = False
            if st.button("Restart Assessment"):
                st.session_state.responses = {}
                st.session_state.current_q = 0
                st.session_state.recommendations_generated = False
                st.session_state.ai_recommendation = None
                st.session_state.scores = None
                st.session_state.top_category = None
                st.rerun()


# --------------------------
# 🚀 Application Entry Point
# --------------------------
def main():
    if not check_auth():
        login()
    else:
        # Add navigation options
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.radio("Choose a mode:",
                                    ["Student Application", "AI Supervisor Chat", "Counselor Dashboard"])

        if app_mode == "Student Application":
            student_application()
        elif app_mode == "AI Supervisor Chat":
            ai_supervisor_chat()
        elif app_mode == "Counselor Dashboard" and st.session_state.get("user_type") == "counselor":
            counselor_dashboard()


if __name__ == "__main__":
    if 'OPENAI_API_KEY' in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    else:
        with st.sidebar:
            st.warning("Enter OpenAI API Key")
            api_key = st.text_input("Key:", type="password")
            if api_key:
                openai.api_key = api_key
                st.success("✓ Ready for AI analysis")

    # Initialize sample data (for demo)
    conn = sqlite3.connect('student_data.db')
    c = conn.cursor()

    # Insert sample case studies if none exist
    c.execute("SELECT COUNT(*) FROM case_studies")
    if c.fetchone()[0] == 0:
        sample_studies = [
            ("STEM Career Transition",
             "A student switched from Humanities to Computer Science after assessment",
             "Landed internship at tech startup", "STEM,career_change"),
            ("Business Major Success",
             "Student with average grades found perfect fit in Marketing",
             "Now working as Digital Marketing Specialist", "Business,non-traditional")
        ]
        c.executemany("INSERT INTO case_studies (title, description, outcome, tags) VALUES (?, ?, ?, ?)",
                      sample_studies)
        conn.commit()

    conn.close()

    main()

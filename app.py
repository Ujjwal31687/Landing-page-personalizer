import io
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import requests
import base64
from backend import app_workflow

# Page configuration for wider layout
st.set_page_config(page_title="Troopod CRO Personalizer", layout="wide")
st.title("Dynamic Landing Page Personalizer")
st.markdown("Match your landing page to your ad creative instantly.")

#Initialize the session state
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# A two column layout
col1, col2 = st.columns([1,1.5])        # number representing the weight of each column

with col1:
    st.subheader("Upload Data")
    #landing page URL
    landing_page_url = st.text_input("Enter URL for the landing page.....")

    st.divider()        # a vertical divider

    # Handling the Ad input
    ad_input_type=st.radio("Ad Creative Type", ["Image upload","Ad Link"])

    uploaded_image=None
    ad_text_url=None

    if ad_input_type == "Image upload":
        uploaded_image = st.file_uploader("Upload Ad Image", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            img_bytes = uploaded_image.read()   # read ONCE

            image = Image.open(io.BytesIO(img_bytes))  # use bytes safely
            st.image(image, caption="Ad Preview", use_container_width=True)

            st.session_state.img_bytes = img_bytes  # store for later

    else:
        ad_text_url = st.text_input("Enter Ad Link URL (e.g. link to an ad image or ad landing page)")
        if ad_text_url:
            st.caption("Ad Preview")
            st.components.v1.html(
                f'<iframe src="{ad_text_url}" style="width:100%;height:300px;border:1px solid #ccc;border-radius:8px;" sandbox="allow-scripts allow-same-origin"></iframe>',
                height=320,
            )

    # Action Button
    if st.button("Generate Personalized Page", type="primary"):
        if landing_page_url and (uploaded_image or ad_text_url):
            with st.status("Running Optimization Pipeline...", expanded=True) as status:
                try:
                    st.write("Initializing LangGraph Workflow...")

                    # Resolve ad creative input
                    base64_img = ""
                    ad_url = ""

                    if uploaded_image:
                        img_bytes = st.session_state.get("img_bytes")
                        if img_bytes:
                            base64_img = base64.b64encode(img_bytes).decode('utf-8')
                    elif ad_text_url:
                        # Check if the URL points directly to an image file
                        if any(ad_text_url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                            # Download and convert to base64 directly
                            st.write("Downloading ad image from URL...")
                            img_resp = requests.get(ad_text_url, timeout=15)
                            img_resp.raise_for_status()
                            base64_img = base64.b64encode(img_resp.content).decode('utf-8')
                        else:
                            # It's a webpage — backend will screenshot it
                            ad_url = ad_text_url

                    st.write("Fetching landing page and running CRO analysis...")
                    from backend import app_workflow

                    initial_state = {
                        "landing_page_url": landing_page_url,
                        "ad_image_base64": base64_img,
                        "ad_url": ad_url,
                        "original_html": "",
                        "raw_html": "",
                        "clean_text": "",
                        "element_map_text": "",
                        "element_lookup": {},
                        "soup_html": "",
                        "ai_results": None,
                        "modified_html": "",
                        "applied_count": 0,
                        "error": None
                    }

                    final_state = app_workflow.invoke(initial_state)

                    if final_state.get("error"):
                        raise Exception(final_state["error"])

                    # Save everything to memory so it doesn't vanish on reload
                    st.session_state.results = final_state["ai_results"]
                    st.session_state.applied_count = final_state["applied_count"]
                    st.session_state.final_modified_html = final_state["modified_html"]
                    st.session_state.original_html = final_state["original_html"]
                    st.session_state.processing_complete = True

                    # Mark the pipeline as finished!
                    status.update(label="Optimization Complete!", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="Pipeline Failed", state="error", expanded=False)
                    st.error(f"Something went wrong: {e}")
                    st.info("Tip: Make sure your GROQ_API_KEY is set in your .env file and the URL is accessible.")
        else:
            st.error("Please provide both a landing page URL and an ad creative")

with col2:
    st.subheader("Results Preview")

    if st.session_state.processing_complete:
        with st.expander("🔍 View Extracted Ad Insights", expanded=True):
            results = st.session_state.get("results")
            applied_count = st.session_state.get("applied_count", 0)

            if results:
                st.write("🎯 **Target Audience:**", results.target_audience)
                st.write("🔥 **Core Offer:**", results.ad_core_offer)
                st.write(f"✅ **Changes Applied:** {applied_count} / {len(results.changes)} successful")
                st.divider()
                for i, c in enumerate(results.changes, 1):
                    st.markdown(f"**Change {i} — `{c.element_type.upper()}`**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption("Before")
                        st.code(c.original_text[:100], language=None)
                    with col_b:
                        st.caption("After")
                        st.code(c.new_text[:100], language=None)
            else:
                st.info("No insights available yet.")
        # Use tabs to compare Original vs Personalized
        tab1, tab2 = st.tabs(["Personalized Output", "Original Page"])
        with tab1:
            st.success("Personalized HTML rendered below.")
            if st.session_state.get("final_modified_html"):
                components.html(st.session_state.final_modified_html, height=800, scrolling=True)
            else:
                st.info("Rendered HTML goes here")
        with tab2:
            st.caption("Original Landing Page")
            if st.session_state.get("original_html"):
                components.html(st.session_state.original_html, height=800, scrolling=True)
            else:
                st.info("Original HTML goes here")
    else:
        st.info("Enter your URLs and hit generate to see the results.")
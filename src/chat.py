from .common import flux_image

with flux_image.imports():
    import os

    from dotenv import load_dotenv
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq


class VisualChain:
    def enter(self):
        print("[VisualChain][__init__] initializing")
        load_dotenv()
        groq_key = os.getenv("GROQ_API_KEY")

        self.vlm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",  # Updated to latest Vision model on Groq
            temperature=0.0,
            api_key=groq_key,
        )

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", temperature=0.3, api_key=groq_key
        )

        self.coach_prompt = ChatPromptTemplate.from_template(
            """
            You are an elite Sports Performance Coach.
            
            INPUT CONTEXT:
            - Exercise: {exercise_name}
            - Technical Analysis: "{observation}"
            
            TASK:
            Translate the technical analysis into a clear, helpful feedback summary.
            
            OUTPUT FORMAT (Strictly follow this layout):
            
            **Correction 1 (Priority):**
            * **The Error:** (Simple English. What are they doing wrong?)
            * **The Fix:** (Specific actionable step)
            * **The Cue:** (2-4 word mental trigger)
            
            **Correction 2 (Secondary):**
            (If no second error, write "None")
            * **The Error:** ...
            * **The Fix:** ...
            * **The Cue:** ...
            """
        )
        self.coach_chain = self.coach_prompt | self.llm | StrOutputParser()

    def analyze_images(self, user_img_b64, trainer_img_b64, exercise_name="Exercise"):
        """
        Takes Base64 images (with skeletons already drawn), gets technical analysis, then converts to coaching cue.
        """
        print(
            f"[VisualChain][analyze_images] analyzing images for exercise: {exercise_name}"
        )
        print(f" VLM analyzing {exercise_name}...")

        vlm_messages = [
            SystemMessage(
                content="You are a Biomechanics Expert. Analyze the skeletal overlays in the images."
            ),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"""
                    TASK: Compare the Biomechanics of the USER (Image 1) vs. the TRAINER (Image 2).
                    
                    CONTEXT: 
                    These images have skeletal lines drawn on them (Green joints, Red lines). 
                    - Image 1: User performing {exercise_name}.
                    - Image 2: Trainer performing {exercise_name} (Perfect Form).

                    INSTRUCTIONS:
                    1. Compare the JOINT ANGLES based on the red lines.
                    2. Look specifically at: Spine Neutrality, Knee Tracking, Hip Depth, Arm Alignment.
                    3. IGNORE background. Focus ONLY on the geometry of the skeleton.

                    OUTPUT:
                    Identify the geometric differences. Be technical (e.g., "User's knee valgus angle is greater").
                    """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{user_img_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{trainer_img_b64}"
                        },
                    },
                ]
            ),
        ]

        try:
            vlm_response = self.vlm.invoke(vlm_messages)
            technical_observation = vlm_response.content

            final_cue = self.coach_chain.invoke(
                {"observation": technical_observation, "exercise_name": exercise_name}
            )

            return final_cue
        except Exception as e:
            print(f"Error in VLM chain: {e}")
            return "Could not generate feedback. Please try again."

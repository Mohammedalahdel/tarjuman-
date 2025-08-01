import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="ترجمان - مترجم لغة الإشارة", layout="centered")

st.title("🤟 مشروع ترجمان")
st.subheader("ترجمة لغة الإشارة إلى نص مكتوب (نسخة تجريبية)")

st.write("قم برفع صورة تحتوي على إشارة يد (ASL)، وسنقوم بتحليلها تلقائيًا.")

# تحميل الصورة من المستخدم
uploaded_file = st.file_uploader("📷 ارفع صورة للإشارة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 الصورة الأصلية", use_column_width=True)

    # تحويل الصورة إلى تنسيق OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # إعداد mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # المعالجة
    results = hands.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # رسم النقاط على الصورة
            annotated_image = img_cv.copy()
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="🧠 تحليل Mediapipe", use_column_width=True)
        st.success("✅ تم التعرف على يد في الصورة! (الترجمة الكاملة قيد التطوير)")
    else:
        st.warning("❌ لم يتم التعرف على يد في الصورة. جرّب صورة أخرى واضحة.")

    hands.close()

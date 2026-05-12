LANGUAGE_NAMES = {"zh": "Chinese", "en": "English", "ru": "Russian", "vi": "Vietnamese"}

LANGUAGE_FIELDS = {
    "zh": {"question": "question", "answer": "answer"},
    "en": {"question": "question_en", "answer": "answer_en"},
    "ru": {"question": "question_ru", "answer": "answer_ru"},
    "vi": {"question": "question_vi", "answer": "answer_vi"},
}

LANGUAGE_PROMPTS = {
    "zh": "你现在是工业品行业专家。现在用户向你提交了一个问题。\n\n问题如下：\n${question}\n\n回复你知道的答案。",
    "en": "You are an industrial domain expert. A user has submitted the following question:\n\nQuestion: ${question}\n\nPlease provide your answer.",
    "ru": "Вы эксперт в промышленной сфере. Пользователь задал вам следующий вопрос:\n\nВопрос: ${question}\n\nПожалуйста, дайте ваш ответ.",
    "vi": "Bạn là chuyên gia trong lĩnh vực công nghiệp. Người dùng đã gửi câu hỏi sau:\n\nCâu hỏi: ${question}\n\nVui lòng cung cấp câu trả lời của bạn.",
}

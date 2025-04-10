"""Модуль сервера для распознавания речи с поддержкой Google Speech Recognition и Whisper.

Сервер предоставляет REST API для:
- Конвертации аудио форматов
- Распознавания речи через Google API
- Распознавания речи через Whisper модели разных размеров
- Управления жизненным циклом моделей Whisper
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import tempfile
import io
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import speech_recognition as sr
from pydub import AudioSegment
import os
import logging
from faster_whisper import WhisperModel

from jiwer import wer
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Конфигурация моделей Whisper
WHISPER_MODELS_CONFIG = {
    "tiny": {"path": "tiny", "device": "cpu", "compute_type": "int8"},
    "base": {"path": "base", "device": "cpu", "compute_type": "int8"},
    "small": {"path": "small", "device": "cpu", "compute_type": "int8"}
}

# Время старта сервера
START_TIME = datetime.now()


def calculate_wer_and_highlight(reference: str, hypothesis: str) -> Tuple[float, str]:
    """Вычисляет Word Error Rate (WER) и создает HTML с подсветкой различий между текстами.
    
    Args:
        reference: Эталонный текст
        hypothesis: Распознанный текст для сравнения
        
    Returns:
        Tuple[float, str]: 
            - WER score (0.0 - полное совпадение, 1.0 - все слова ошибочны)
            - HTML строка с подсветкой различий цветами:
                * Зеленый: правильные слова
                * Красный: замененные слова
                * Подчеркивание: лишние слова
                * Зачеркивание: пропущенные слова

    Raises:
        ValueError: Если оба текста пустые
    """
    logger.debug("Начало вычисления WER для текстов длиной %d и %d символов", 
                len(reference), len(hypothesis))
    
    # Проверка пустых входных данных
    if not reference and not hypothesis:
        error_msg = "Оба текста (reference и hypothesis) не могут быть пустыми"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Вычисление WER
    error_rate = wer(reference, hypothesis)
    
    # Генерация подсветки различий
    matcher = SequenceMatcher(None, reference.split(), hypothesis.split())
    html_parts = []
    
    # Обработка различных типов различий
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for word in hyp_words[j1:j2]:
                html_parts.append(f'<span style="color: green;">{word}</span>')
        elif tag == 'replace':
            for word in hyp_words[j1:j2]:
                html_parts.append(f'<span style="color: red;">{word}</span>')
        elif tag == 'insert':
            for word in hyp_words[j1:j2]:
                html_parts.append(
                    f'<span style="color: red; text-decoration: underline;">{word}</span>')
        elif tag == 'delete':
            for word in ref_words[i1:i2]:
                html_parts.append(
                    f'<span style="color: red; text-decoration: line-through;">{word}</span>')

    highlighted_html = ' '.join(html_parts)
    
    logger.info("WER вычислен успешно. Score: %.2f", error_rate)
    return error_rate, highlighted_html


class WhisperModelManager:
    """Менеджер для загрузки, хранения и выгрузки моделей Whisper.

    Реализует паттерн Singleton и обеспечивает:
    - Ленивую загрузку моделей
    - Автоматическую выгрузку неиспользуемых моделей
    - Потокобезопасный доступ к моделям

    Attributes:
        _instance: Единственный экземпляр менеджера (Singleton)
        _lock: Блокировка для обеспечения потокобезопасности
        _models: Словарь загруженных моделей {model_size: (model, last_used_time)}
        _cleanup_interval: Интервал проверки неактивных моделей в секундах
        _inactivity_timeout: Время неактивности для выгрузки модели в секундах
    """

    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Tuple[WhisperModel, datetime]] = {}
    _cleanup_interval = 300  # 5 минут
    _inactivity_timeout = 3600  # 1 час

    def __new__(cls):
        """Реализация паттерна Singleton.

        Returns:
            WhisperModelManager: Единственный экземпляр менеджера
        """
        if cls._instance is None:
            cls._instance = super(WhisperModelManager, cls).__new__(cls)
            cls._start_cleanup_thread()
        return cls._instance

    @classmethod
    def _start_cleanup_thread(cls):
        """Запускает фоновый поток для очистки неактивных моделей."""
        def cleanup():
            while True:
                time.sleep(cls._cleanup_interval)
                cls._cleanup_inactive_models()

        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()

    @classmethod
    def _cleanup_inactive_models(cls):
        """Выгружает модели, которые не использовались дольше _inactivity_timeout."""
        with cls._lock:
            current_time = datetime.now()
            models_to_remove = [
                model_size for model_size, (_, last_used) in cls._models.items()
                if (current_time - last_used).total_seconds() > cls._inactivity_timeout
            ]

            for model_size in models_to_remove:
                logger.info(
                    f"Выгружаем модель Whisper {model_size} из-за неактивности")
                del cls._models[model_size]

    @classmethod
    def get_model(cls, model_size: str) -> Optional[WhisperModel]:
        """Возвращает модель Whisper для указанного размера.

        При первом обращении загружает модель. Обновляет время последнего использования.

        Args:
            model_size: Размер модели ('tiny', 'base', 'small')

        Returns:
            WhisperModel: Загруженная модель или None в случае ошибки
        """
        with cls._lock:
            if model_size not in cls._models:
                if model_size not in WHISPER_MODELS_CONFIG:
                    logger.error(f"Неизвестный размер модели: {model_size}")
                    return None

                logger.info(f"Загружаем модель Whisper {model_size}")
                try:
                    config = WHISPER_MODELS_CONFIG[model_size]
                    model = WhisperModel(
                        config["path"],
                        device=config["device"],
                        compute_type=config["compute_type"]
                    )
                    cls._models[model_size] = (model, datetime.now())
                except Exception as e:
                    logger.error(
                        f"Ошибка загрузки модели {model_size}: {str(e)}", exc_info=True)
                    return None

            model, _ = cls._models[model_size]
            cls._models[model_size] = (model, datetime.now())
            return model


class AudioProcessor:
    """Класс для обработки и конвертации аудио данных."""

    @staticmethod
    def convert_webm_to_wav(webm_data: bytes) -> Optional[bytes]:
        """Конвертирует аудио данные из формата WebM в WAV.

        Args:
            webm_data: Байты аудио в формате WebM

        Returns:
            bytes: Байты аудио в формате WAV или None в случае ошибки

        Note:
            Выходной формат:
            - Частота дискретизации: 16 kHz
            - Каналы: 1 (моно)
            - Размер сэмпла: 16 бит
        """
        try:
            audio = AudioSegment.from_file(
                io.BytesIO(webm_data),
                format="webm",
                codec="opus"
            )

            audio = audio.set_frame_rate(
                16000).set_channels(1).set_sample_width(2)

            buffer = io.BytesIO()
            audio.export(
                buffer,
                format="wav",
                codec="pcm_s16le",
                parameters=["-ar", "16000"]
            )
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Ошибка конвертации: {str(e)}", exc_info=True)
            return None


class SpeechRecognizer:
    """Класс для распознавания речи разными методами."""

    @staticmethod
    def recognize_with_google(wav_data: bytes, language: str = 'ru-RU') -> Tuple[Optional[str], Optional[str]]:
        """Распознаёт речь с помощью Google Speech Recognition API.

        Args:
            wav_data: Байты аудио в формате WAV
            language: Язык распознавания (по умолчанию 'ru-RU')

        Returns:
            Tuple: (распознанный текст, ошибка) - одно из значений всегда None
        """
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(wav_data)) as source:
                audio_data = recognizer.record(source)
                logger.debug("Начало распознавания...")
                text = recognizer.recognize_google(
                    audio_data, language=language)
                return text, None
        except sr.UnknownValueError:
            error = "Не удалось распознать речь"
            logger.warning(error)
            return None, error
        except sr.RequestError as e:
            error = f"Ошибка API: {e}"
            logger.error(error)
            return None, error
        except Exception as e:
            error = f"Ошибка распознавания: {str(e)}"
            logger.error(error, exc_info=True)
            return None, error

    @staticmethod
    def recognize_with_whisper(wav_data: bytes, model_size: str) -> Tuple[Optional[str], Optional[str]]:
        """Распознаёт речь с помощью Whisper модели указанного размера.

        Args:
            wav_data: Байты аудио в формате WAV
            model_size: Размер модели ('tiny', 'base', 'small')

        Returns:
            Tuple: (распознанный текст, ошибка) - одно из значений всегда None

        Note:
            Создаёт временный файл для работы Whisper, который автоматически удаляется
        """
        try:
            model = WhisperModelManager.get_model(model_size)
            if model is None:
                error = f"Не удалось загрузить модель {model_size}"
                logger.error(error)
                return None, error

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                temp_wav.write(wav_data)
                temp_wav.flush()

                segments, _ = model.transcribe(temp_wav.name)
                text = " ".join(segment.text for segment in segments)
                return text, None
        except Exception as e:
            error = f"Ошибка Whisper: {str(e)}"
            logger.error(error, exc_info=True)
            return None, error


@app.route('/health', methods=['GET'])
def health_check():
    """Базовый эндпоинт проверки работоспособности"""
    return jsonify({
        "status": "OK",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/status', methods=['GET'])
def full_status():
    """Расширенный статус системы"""
    status = {
        "status": "OK",
        "version": "1.0.0",
        "uptime": str(datetime.now() - START_TIME),
        "database_connected": False,
        "timestamp": datetime.now().isoformat()
    }

    # Проверка подключения к БД
    try:
        with engine.connect() as conn:
            conn.execute(db.text("SELECT 1"))
            status['database_connected'] = True
    except Exception as e:
        status['database_status'] = str(e)

    return jsonify(status), 200


@app.route('/')
def home():
    """Корневой эндпоинт"""
    return jsonify({
        "message": "API is running",
        "documentation": "/swagger-ui"  # Если используется Swagger
    }), 200


@app.route('/favicon.ico')
def favicon():
    """Отдаёт favicon для браузера.

    Returns:
        Response: Иконка сайта
    """
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )


@app.after_request
def add_headers(response):
    """Добавляет стандартные заголовки к каждому ответу.

    Args:
        response: Оригинальный response

    Returns:
        Response: Модифицированный response с заголовками
    """
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


@app.route('/')
def index():
    """Главная страница сервера.

    Returns:
        str: HTML главной страницы
    """
    return render_template('index.html')


def handle_audio_upload() -> Tuple[Optional[bytes], Optional[str]]:
    """Обрабатывает загрузку аудиофайла из запроса.

    Returns:
        Tuple: (данные файла, ошибка) - одно из значений всегда None

    Raises:
        HTTP 400: Если файл не найден, пустой или слишком большой
    """
    if 'audio' not in request.files:
        return None, 'Файл не найден'

    file = request.files['audio']
    if file.filename == '':
        return None, 'Пустое имя файла'

    try:
        file_data = file.read()
        if len(file_data) > app.config['MAX_CONTENT_LENGTH']:
            return None, 'Файл слишком большой'
        return file_data, None
    except Exception as e:
        error = f"Ошибка чтения файла: {str(e)}"
        logger.error(error)
        return None, error


@app.route('/recognize', methods=['POST'])
def recognize():
    """API endpoint для распознавания речи через Google Speech Recognition.

    Returns:
        JSON: {'text': распознанный текст} или {'error': сообщение об ошибке}

    Raises:
        HTTP 400: Ошибки клиента (неверный запрос)
        HTTP 500: Ошибки сервера
    """
    logger.debug("Получен запрос /recognize")

    file_data, error = handle_audio_upload()
    if error:
        return jsonify({'error': error}), 400

    wav_data = AudioProcessor.convert_webm_to_wav(file_data)
    if not wav_data:
        return jsonify({'error': 'Ошибка конвертации аудио'}), 400

    text, error = SpeechRecognizer.recognize_with_google(wav_data)
    if error:
        return jsonify({'error': error}), 400 if "Не удалось распознать речь" in error else 500

    logger.info("Результат распознавания: %s", text)
    return jsonify({'text': text})


@app.route('/recognize_whisper/<model_size>', methods=['POST'])
def recognize_whisper(model_size: str):
    """API endpoint для распознавания речи через Whisper.

    Args:
        model_size: Размер модели ('tiny', 'base', 'small')

    Returns:
        JSON: {'text': распознанный текст} или {'error': сообщение об ошибке}

    Raises:
        HTTP 400: Ошибки клиента (неверный запрос или неизвестная модель)
        HTTP 500: Ошибки сервера
    """
    logger.debug(f"Получен запрос /recognize_whisper с моделью {model_size}")

    file_data, error = handle_audio_upload()
    if error:
        return jsonify({'error': error}), 400

    wav_data = AudioProcessor.convert_webm_to_wav(file_data)
    if not wav_data:
        return jsonify({'error': 'Ошибка конвертации аудио'}), 400

    text, error = SpeechRecognizer.recognize_with_whisper(wav_data, model_size)
    if error:
        return jsonify({'error': error}), 400 if "Неизвестный размер модели" in error else 500

    logger.info("Результат распознавания Whisper (%s): %s", model_size, text)
    return jsonify({'text': text})


@app.route('/evaluate_wer', methods=['POST'])
def evaluate_text():
    """API endpoint для оценки качества распознавания речи с помощью WER.
    
    Принимает JSON с полями:
    {
        "reference": "Эталонный текст",
        "hypothesis": "Распознанный текст"
    }
    
    Returns:
        JSON: {
            "wer_score": float, 
            "highlighted_html": str (HTML с подсветкой)
        } или {"error": "сообщение"}
        
    Пример ошибок:
        - 400: Неверный формат запроса или отсутствуют поля
        - 500: Внутренняя ошибка при обработке
    """
    logger.info("Получен запрос на оценку WER")
    
    try:
        data = request.get_json()
        logger.debug("Данные запроса: %s", data)
        
        # Валидация входных данных
        if not data or 'reference' not in data or 'hypothesis' not in data:
            error_msg = "Отсутствуют обязательные поля: reference и hypothesis"
            logger.warning(error_msg)
            return jsonify({'error': error_msg}), 400
            
        reference = str(data['reference'])
        hypothesis = str(data['hypothesis'])
        
        logger.debug("Обработка текстов: reference(%d chars), hypothesis(%d chars)",
                    len(reference), len(hypothesis))
        
        # Вычисление WER и подсветки
        wer_score, highlighted_html = calculate_wer_and_highlight(reference, hypothesis)
        
        logger.info("Успешный результат WER: %.2f", wer_score)
        return jsonify({
            'wer_score': wer_score,
            'highlighted_html': highlighted_html
        })
        
    except ValueError as e:
        error_msg = f"Ошибка валидации: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    except Exception as e:
        error_msg = f"Непредвиденная ошибка: {str(e)}"
        logger.exception(error_msg)
        return jsonify({'error': error_msg}), 500


if __name__ == '__main__':
    """Точка входа для запуска сервера."""
    app.run(host='0.0.0.0', port=5000, debug=False)

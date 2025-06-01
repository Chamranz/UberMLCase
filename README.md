Модель из ключевого кейса обернута в fastapi. В шаблоне cookie-cutter. Чтобы собрать репозиторий по шаблону:

1. pip install cookiecutter

2. cookiecutter https://github.com/Chamranz/UberMLCase.git - проставляем нужные параметры проекта

Для удобства ввода данных имеется форма:

👉 http://localhost:8000/predict/form/ - можно протестировать вводные данные и предсказание цены. Может быть удобным в модуле с ML кейсом.

Под капотом градиентный бустинг. JSON посылка с входными данными из сервиса уже переводится в нужный формат признаков/

Kind reminder:

python -m venv .mlops-venv

source .mlops-venv/bin/activate

pip install -r requirements.txt

Аминь

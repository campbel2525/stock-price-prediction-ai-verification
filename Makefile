pn = 'stock-price-prediction-ai-verification'

init: ## 開発作成
	docker compose -p $(pn) build --no-cache
	docker compose -p $(pn) down --volumes
	docker compose -p $(pn) up -d
	make install

up: ## 開発立ち上げ
	docker compose -p $(pn) up -d

down: ## 開発down
	docker compose -p $(pn) down

shell: ## dockerのshellに入る
	docker compose -p $(pn) exec app bash

check: ## コードのフォーマット
	docker compose -p $(pn) exec -it app pipenv run isort .
	docker compose -p $(pn) exec -it app pipenv run black .
	docker compose -p $(pn) exec -it app pipenv run flake8 .
	docker compose -p $(pn) exec -it app pipenv run mypy .

install:
	docker compose -p $(pn) exec -it app pipenv install --dev

destroy: ## 環境削除
	make down
	docker network ls -qf name=$(pn) | xargs docker network rm
	docker container ls -a -qf name=$(pn) | xargs docker container rm
	docker volume ls -qf name=$(pn) | xargs docker volume rm

include .env.docker

pn := $(PROJECT_NAME)
user_name := $(USER_NAME)
user_group := $(USER_GROUP)
pf := $(COMPOSE_FILE)

init: ## 開発作成
	make chown
	docker compose -f ${pf} -p $(pn) build --no-cache
	docker compose -f ${pf} -p $(pn) down --volumes
	docker compose -f ${pf} -p $(pn) up -d
	make install
	make chown

up: ## 開発立ち上げ
	docker compose -f ${pf} -p $(pn) up -d

down: ## 開発down
	docker compose -f ${pf} -p $(pn) down

shell: ## dockerのshellに入る
	docker compose -f ${pf} -p $(pn) exec app bash

check: ## コードのフォーマット
	docker compose -f ${pf} -p $(pn) exec -it app pipenv run isort .
	docker compose -f ${pf} -p $(pn) exec -it app pipenv run black .
	docker compose -f ${pf} -p $(pn) exec -it app pipenv run flake8 .
	docker compose -f ${pf} -p $(pn) exec -it app pipenv run mypy .

install:
	docker compose -f ${pf} -p $(pn) exec -it app pipenv install --dev

destroy: ## 開発環境削除
	make down
	if [ -n "$(docker network ls -qf name=$(pn))" ]; then \
		docker network ls -qf name=$(pn) | xargs docker network rm; \
	fi
	if [ -n "$(docker container ls -a -qf name=$(pn))" ]; then \
		docker container ls -a -qf name=$(pn) | xargs docker container rm; \
	fi
	if [ -n "$(docker volume ls -qf name=$(pn))" ]; then \
		docker volume ls -qf name=$(pn) | xargs docker volume rm; \
	fi

push:
	git add .
	git commit -m "Commit at $$(date +'%Y-%m-%d %H:%M:%S')"
	git push origin main

# すべてのファイルの所有者を指定したユーザーに変更する
# .env.dockerのUSER_NAMEが指定されている場合に実行
chown:
	if [ -n "${user_name}" ]; then \
		sudo chown -R "${user_name}:${user_group}" ./ ; \
	fi

FROM node:14

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm install

COPY ./src ./src
COPY ./public ./public

EXPOSE 3000

CMD ["npm", "run", "start"]

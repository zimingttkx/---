FROM ubuntu:latest
LABEL authors="apple"

ENTRYPOINT ["top", "-b"]
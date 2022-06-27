REPO ?= ijcai2022nmmo/submission-runtime

TENCENTCLOUD_REGISTRY = ccr.ccs.tencentyun.com

TAG ?= $(shell date +%Y%m%d)-$(shell git describe --match=$(git rev-parse --short=8 HEAD) --tags --always --dirty)

.PHONY : image push

image:
	@docker build -t ${REPO}:${TAG} -f Dockerfile .
	@docker tag ${REPO}:${TAG} ${REPO}:latest

push: image
	@docker push ${REPO}:${TAG}
	@docker push ${REPO}:latest
	@docker tag ${REPO}:${TAG} ${TENCENTCLOUD_REGISTRY}/${REPO}:${TAG}
	@docker push ${TENCENTCLOUD_REGISTRY}/${REPO}:${TAG}
	@docker tag ${REPO}:${TAG} ${TENCENTCLOUD_REGISTRY}/${REPO}:latest
	@docker push ${TENCENTCLOUD_REGISTRY}/${REPO}:latest

BUILD_TYPE?=Release
BUILD_PATH?=$(CURDIR)/../build
TARGET_ROOT_OUTPUT_DIR?=$(CURDIR)/../build/test

TESTS:=$(shell cat $(BUILD_PATH)/.tests)
RESULTS:=$(addsuffix .xml, $(TESTS))

define passed_xmlstr
<?xml version="1.0" encoding="UTF-8"?> \
<testsuites tests="1" failures="0" disabled="0" errors="0" time="0" timestamp="%s" name="AllTests"> \
	<testsuite name="%s" tests="1" failures="0" disabled="0" skipped="0" errors="0" time="0" timestamp="%s"> \
		<testcase name="%s" status="run" result="completed" time="0" timestamp="%s" classname="%s"> \
		</testcase> \
	</testsuite> \
</testsuites>
endef

define failed_xmlstr
<?xml version="1.0" encoding="UTF-8"?> \
<testsuites tests="1" failures="1" disabled="0" errors="0" time="0" timestamp="%s" name="AllTests"> \
	<testsuite name="%s" tests="1" failures="1" disabled="0" skipped="0" errors="0" time="0" timestamp="%s"> \
		<testcase name="%s" status="run" result="completed" time="0" timestamp="%s" classname="%s"> \
			<failure><![CDATA[mismatch]]></failure> \
		</testcase> \
	</testsuite> \
</testsuites>
endef

all: $(RESULTS)
	:

%.xml: prepare
	CASE=$*; \
	FILTER="$${CASE#*.}"; \
	TEST_SUITE="$${FILTER%.*}"; \
	TEST_CASE="$${FILTER#*.}"; \
	BIN_FILE=`find $(TARGET_ROOT_OUTPUT_DIR) -name $${CASE}`; \
	$${BIN_FILE}; \
	RESULT="$$?"; \
	TIMESTAMP=`date "+%Y-%m-%dT%H:%M:%S"`; \
	flock $(BUILD_PATH)/.results printf "%-60s %d\n" "$${CASE}" $${RESULT} >> $(BUILD_PATH)/.results; \
	if [ $${RESULT} = "0" ]; then \
		printf '$(passed_xmlstr)\n' "$${TIMESTAMP}" "$${TEST_SUITE}" "$${TIMESTAMP}" "$${TEST_SUITE}" "$${TIMESTAMP}" "$${TEST_CASE}" > ${BUILD_PATH}/ci_results/$@; \
	else \
		printf '$(failed_xmlstr)\n' "$${TIMESTAMP}" "$${TEST_SUITE}" "$${TIMESTAMP}" "$${TEST_SUITE}" "$${TIMESTAMP}" "$${TEST_CASE}" > ${BUILD_PATH}/ci_results/$@; \
	fi

prepare:
	echo > $(BUILD_PATH)/.results
	rm -rf $(BUILD_PATH)/ci_results || true
	mkdir $(BUILD_PATH)/ci_results

#=======================================================================================
# Copyright (C) 2022 NHR@FAU, University Erlangen-Nuremberg.
# All rights reserved.
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE file.
#=======================================================================================

#CONFIGURE BUILD SYSTEM
TARGET	   = exe-$(TAG)
BUILD_DIR  = ./$(TAG)
SRC_DIR    = ./src
MAKE_DIR   = ./
Q         ?= @

#DO NOT EDIT BELOW
include $(MAKE_DIR)/config.mk
include $(MAKE_DIR)/include_$(TAG).mk
INCLUDES  += -I$(SRC_DIR) -I$(BUILD_DIR)

VPATH     = $(SRC_DIR)
SRC       = $(wildcard $(SRC_DIR)/*.c)
ASM       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.s, $(SRC))
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC))
SOURCES   = $(SRC) $(wildcard $(SRC_DIR)/*.h)
CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(OPTIONS) $(INCLUDES)

${TARGET}: $(BUILD_DIR) $(OBJ)
	$(info ===>  LINKING  $(TARGET))
	$(Q)${LINKER} ${LFLAGS} -o $(TARGET) $(OBJ) $(LIBS)

$(BUILD_DIR)/%.o:  %.c $(MAKE_DIR)/include_$(TAG).mk $(MAKE_DIR)/config.mk
	$(info ===>  COMPILE  $@)
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $< -o $@
	$(Q)$(GCC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.s:  %.c
	$(info ===>  GENERATE ASM  $@)
	$(CC) -S $(CPPFLAGS) $(CFLAGS) $< -o $@

.PHONY: clean distclean tags info asm format

clean:
	$(info ===>  CLEAN)
	@rm -rf $(BUILD_DIR)
	@rm -f tags

distclean: clean
	$(info ===>  DIST CLEAN)
	@rm -f $(TARGET)

info:
	$(info $(CFLAGS))
	$(Q)$(CC) $(VERSION)

asm:  $(BUILD_DIR) $(ASM)

tags:
	$(info ===>  GENERATE TAGS)
	$(Q)ctags -R

format:
	@for src in $(SOURCES) ; do \
		echo "Formatting $$src" ; \
		clang-format -i $$src ; \
	done
	@echo "Done"

$(BUILD_DIR):
	@mkdir $(BUILD_DIR)

-include $(OBJ:.o=.d)

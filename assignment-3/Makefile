#=======================================================================================
# Copyright (C) NHR@FAU, University Erlangen-Nuremberg.
# All rights reserved.
# Use of this source code is governed by a MIT-style
# license that can be found in the LICENSE file.
#=======================================================================================

#CONFIGURE BUILD SYSTEM
TARGET	   = exe-$(TAG)
BUILD_DIR  = ./build/$(TAG)
SRC_DIR    = ./src
MAKE_DIR   = ./
Q         ?= @

#DO NOT EDIT BELOW
include $(MAKE_DIR)/config.mk
include $(MAKE_DIR)/include_$(TAG).mk
INCLUDES  += -I$(SRC_DIR)/includes -I$(BUILD_DIR)

VPATH     = $(SRC_DIR)
ASM       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.s,$(wildcard $(SRC_DIR)/*.c))
OBJ       = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
CPPFLAGS := $(CPPFLAGS) $(DEFINES) $(OPTIONS) $(INCLUDES)

${TARGET}: $(BUILD_DIR) $(OBJ)
	$(info ===>  LINKING  $(TARGET))
	$(Q)${LINKER} ${LFLAGS} -o $(TARGET) $(OBJ) $(LIBS)

$(BUILD_DIR)/%.o:  %.c $(MAKE_DIR)/include_$(TAG).mk
	$(info ===>  COMPILE  $@)
	$(CC) -c $(CPPFLAGS) $(CFLAGS) $< -o $@
	$(Q)$(GCC) $(CPPFLAGS) -MT $(@:.d=.o) -MM  $< > $(BUILD_DIR)/$*.d

$(BUILD_DIR)/%.s:  %.c
	$(info ===>  GENERATE ASM  $@)
	$(CC) -S $(CPPFLAGS) $(CFLAGS) $< -o $@

.PHONY: clean distclean tags info asm

clean:
	$(info ===>  CLEAN)
	@rm -rf $(BUILD_DIR)
	@rm -f tags

distclean: clean
	$(info ===>  DIST CLEAN)	
	@rm -rf build

	@rm -f $(TARGET)

info:
	$(info $(CFLAGS))
	$(Q)$(CC) $(VERSION)

asm:  $(BUILD_DIR) $(ASM)

tags:
	$(info ===>  GENERATE TAGS)
	$(Q)ctags -R

format:
	@for src in $(SRC) ; do \
		echo "Formatting $$src" ; \
		clang-format -i $$src ; \
	done
	@echo "Done"
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
.clangd:
	$(file > .clangd,$(CLANGD_TEMPLATE))

-include $(OBJ:.o=.d)

MS:= \
    tao-cv

all: $(addsuffix _ms,$(MS))

%_ms:
	$(MAKE) -C $(@:_ms=)

clean:
	for dir in $(MS); do $(MAKE) -C $$dir clean; done


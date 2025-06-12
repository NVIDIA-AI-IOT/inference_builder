    async def {{ name }}(self, request, body, **kwargs):
        accept = request.headers.get("accept", "")
        streaming = "application/x-ndjson" in accept
        in_data = self.process_request("{{ name }}", body)
        self.logger.debug(f"request processed as {in_data}")
        try:
            if streaming:
                # If streaming, yield results as they are processed
                async def generate_stream():
                    async with self._lock:
                        async for result in self._inference.execute(in_data):
                            response = self.process_response("infer", request, result)
                            yield response

                # Wrap the generator with StreamingResponse
                return 200, StreamingResponse(generate_stream(), media_type="application/x-ndjson")
            else:
                # If not streaming, process and return the last result
                response = None
                async with self._lock:
                    async for result in self._inference.execute(in_data):
                        response = self.process_response("infer", request, result)
                    return (200, response) if response else (500, "No results generated from inference")
        except Exception as e:
            self.logger.error(f"Inference failed: {type(e).__name__}: {e}")
            return 500, str(e)

    async def {{ name }}(self, interface, request, body, *args):
        accept = request.headers.get("accept", "")
        streaming = "application/x-ndjson" in accept
        in_data = self.process_request("{{ name }}", body)
        self.logger.debug(f"request processed as {in_data}")
        try:
            if streaming:
                # If streaming, yield results as they are processed
                async def generate_stream():
                    async for result in self._inference.execute(in_data):
                        response = self.process_response("infer", request, result)
                        yield response

                # Wrap the generator with StreamingResponse
                return 200, StreamingResponse(generate_stream(), media_type="application/x-ndjson")
            else:
                # If not streaming, process and return the last result
                async for result in self._inference.execute(in_data):
                    response = self.process_response("infer", request, result)
                return 200, response
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return 500, str(e)

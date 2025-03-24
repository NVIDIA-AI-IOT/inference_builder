    async def {{ name }}(self, interface, request, body):
        self.logger.info("{{ name }} called")
        in_data = self.process_request("{{ name }}", body)
        self.logger.debug(f"request processed as {in_data}")
        result =  await self._inference.execute(in_data)
        response = self.process_response("{{ name }}", body, result)
        self.logger.debug(f"response generated as {response}")
        return response

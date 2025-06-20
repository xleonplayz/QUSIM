export function onParamChange(param: string, value: number) {
  setSimParams({ ...simParams, [param]: value });
  simulateAndRender(simParams);
}
